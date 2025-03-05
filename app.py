import streamlit as st
import os
import json
import logging
import time
from google.cloud import texttospeech
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
    VideoFileClip,
    CompositeVideoClip,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)

# Cargar credenciales de GCP. Comentado para ejecución local, descomentar para Streamlit Cloud.
# credentials = dict(st.secrets["gcp_service_account"])
# with open("google_credentials.json", "w") as f:
#     json.dump(credentials, f)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"


# Configuración de voces
VOCES_DISPONIBLES = {
    'es-ES-Standard-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Standard-A': texttospeech.SsmlVoiceGender.FEMALE,
}


def create_text_image(text, size=(1280, 360), font_size=30, line_height=40):
    img = Image.new('RGB', size, 'black')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size) # Asegúrate de que la ruta sea correcta

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        if right > size[0] - 60:  # Margen
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    total_height = len(lines) * line_height
    y = (size[1] - total_height) // 2  # Centrado vertical

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (right - left)) // 2  # Centrado horizontal
        draw.text((x, y), line, font=font, fill="white")
        y += line_height

    return np.array(img)


def loop_video(clip, duration):
    """Loops a video clip to a specified duration."""
    clips = []
    remaining_duration = duration
    while remaining_duration > 0:
        if remaining_duration < clip.duration:
            clips.append(clip.subclip(0, remaining_duration))
            remaining_duration = 0
        else:
            clips.append(clip)
            remaining_duration -= clip.duration
    final_clip = concatenate_videoclips(clips, method="compose")
    return final_clip


def create_simple_video(texto, nombre_salida, voz, video_fondo):
    archivos_temp = []
    clips_audio = []
    clips_finales = []
    video_fondo_clip = None

    try:
        logging.info("Iniciando proceso de creación de video...")
        frases = [f.strip() + "." for f in texto.split('.') if f.strip()]
        client = texttospeech.TextToSpeechClient()

        tiempo_acumulado = 0

        segmentos_texto = []
        segmento_actual = ""
        for frase in frases:
            if len(segmento_actual) + len(frase) < 300:
                segmento_actual += " " + frase
            else:
                segmentos_texto.append(segmento_actual.strip())
                segmento_actual = frase
        segmentos_texto.append(segmento_actual.strip())

        for i, segmento in enumerate(segmentos_texto):
            logging.info(f"Procesando segmento {i+1} de {len(segmentos_texto)}")

            synthesis_input = texttospeech.SynthesisInput(text=segmento)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES", name=voz, ssml_gender=VOCES_DISPONIBLES[voz]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    response = client.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )
                    break
                except Exception as e:
                    logging.error(f"Error al solicitar audio (intento {retry_count + 1}): {str(e)}")
                    if "429" in str(e):  # Rate Limit
                        retry_count += 1
                        time.sleep(2**retry_count)  # Exponential backoff
                    else:
                        raise

            if retry_count > max_retries:
                raise Exception("Máximos intentos de reintento alcanzados")

            temp_filename = f"temp_audio_{i}.mp3"
            archivos_temp.append(temp_filename)
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)

            audio_clip = AudioFileClip(temp_filename)
            clips_audio.append(audio_clip)
            duracion = audio_clip.duration

            text_img = create_text_image(segmento)
            txt_clip = (
                ImageClip(text_img)
                .set_start(tiempo_acumulado)
                .set_duration(duracion)
                .set_position('center')
            )

            video_segment = txt_clip.set_audio(audio_clip.set_start(tiempo_acumulado))
            clips_finales.append(video_segment)

            tiempo_acumulado += duracion
            time.sleep(0.2)  # Pequeña pausa

        # Cargar el video de fondo
        video_fondo_clip = VideoFileClip(video_fondo)
        video_duracion_total = tiempo_acumulado
        video_fondo_clip = loop_video(video_fondo_clip, video_duracion_total)
        video_fondo_clip = video_fondo_clip.resize((1280, 720))

        video_final = CompositeVideoClip([video_fondo_clip] + clips_finales)

        video_final.write_videofile(
            nombre_salida,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4,
        )
        video_final.close()

        for clip in clips_audio:
            clip.close()
        for clip in clips_finales:
            clip.close()
        if video_fondo_clip:
            video_fondo_clip.close()

        # Limpieza de archivos temporales
        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.error(f"Error al eliminar archivo temporal {temp_file}: {e}")

        return True, "Video generado exitosamente"

    except Exception as e:
        logging.error(f"Error en create_simple_video: {str(e)}")

        # Limpieza en caso de error
        for clip in clips_audio:
            try:
                clip.close()
            except:
                pass
        for clip in clips_finales:
            try:
                clip.close()
            except:
                pass
        if video_fondo_clip:
            try:
                video_fondo_clip.close()
            except:
                pass
        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        return False, str(e)


def main():
    st.title("Creador de Videos Automático")

    uploaded_file = st.file_uploader("Carga un archivo de texto", type="txt")
    voz_seleccionada = st.selectbox("Selecciona la voz", options=list(VOCES_DISPONIBLES.keys()))
    video_fondo = st.file_uploader("Carga un video de fondo (mp4)", type=["mp4"])

    if uploaded_file and video_fondo:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_fondo.read())
            temp_video_path = temp_video.name

        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")

        if st.button("Generar Video"):
            with st.spinner('Generando video...'):
                nombre_salida_completo = f"{nombre_salida}.mp4"
                success, message = create_simple_video(texto, nombre_salida_completo, voz_seleccionada, temp_video_path)
                if success:
                    st.success(message)
                    st.video(nombre_salida_completo)
                    with open(nombre_salida_completo, 'rb') as file:
                        st.download_button(label="Descargar video", data=file, file_name=nombre_salida_completo)
                    st.session_state.video_path = nombre_salida_completo
                else:
                    st.error(f"Error al generar video: {message}")

            os.unlink(temp_video_path)

if __name__ == "__main__":
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    main()
