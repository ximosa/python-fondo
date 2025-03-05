import streamlit as st
import os
import json
import logging
import time
from google.cloud import texttospeech
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)

# Cargar credenciales de GCP desde secrets
try:
    credentials = dict(st.secrets.gcp_service_account)
    with open("google_credentials.json", "w") as f:
        json.dump(credentials, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
except Exception as e:
    st.error(f"Error al cargar las credenciales de GCP: {e}")
    st.stop()

# Configuración de voces
VOCES_DISPONIBLES = {
    'es-ES-Standard-B': texttospeech.SsmlVoiceGender.MALE,
}

# Función de creación de texto
def create_text_image(text, size=(1280, 360), font_size=30, line_height=40):
    img = Image.new('RGBA', size, (0,0,0,0)) # Fondo transparente
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        if right > size[0] - 60:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    total_height = len(lines) * line_height
    y = (size[1] - total_height) // 2

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (right - left)) // 2
        draw.text((x, y), line, font=font, fill="white")
        y += line_height

    return np.array(img)

# Función de creación de video
def create_simple_video(texto, nombre_salida, voz, background_video_path):
    archivos_temp = []
    clips_audio = []
    clips_finales = []
    success = False # Inicializa success
    message = "" # Inicializa message
    temp_video_path = None

    try:
        logging.info("Iniciando proceso de creación de video...")
        frases = [f.strip() + "." for f in texto.split('.') if f.strip()]
        client = texttospeech.TextToSpeechClient()
        
        tiempo_acumulado = 0
        
        # Agrupamos frases en segmentos
        segmentos_texto = []
        segmento_actual = ""
        for frase in frases:
          if len(segmento_actual) + len(frase) < 300:
            segmento_actual += " " + frase
          else:
            segmentos_texto.append(segmento_actual.strip())
            segmento_actual = frase
        segmentos_texto.append(segmento_actual.strip())
        
        # Cargar video de fondo
        try:
            logging.info(f"Intentando cargar video de fondo desde: {background_video_path}")
            background_clip = VideoFileClip(background_video_path, audio=False)
            logging.info("Video de fondo cargado exitosamente.")
        except Exception as e:
            message = f"Error al cargar el video de fondo: {e}"
            logging.error(message)
            return False, message, None

        
        for i, segmento in enumerate(segmentos_texto):
            logging.info(f"Procesando segmento {i+1} de {len(segmentos_texto)}")
            
            synthesis_input = texttospeech.SynthesisInput(text=segmento)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=voz,
                ssml_gender=VOCES_DISPONIBLES[voz]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            retry_count = 0
            max_retries = 3
            
            while retry_count <= max_retries:
              try:
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                break
              except Exception as e:
                  logging.error(f"Error al solicitar audio (intento {retry_count + 1}): {str(e)}")
                  if "429" in str(e):
                    retry_count +=1
                    time.sleep(2**retry_count)
                  else:
                    raise
            
            if retry_count > max_retries:
                raise Exception("Maximos intentos de reintento alcanzado")
            
            temp_filename = f"temp_audio_{i}.mp3"
            archivos_temp.append(temp_filename)
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)
            
            audio_clip = AudioFileClip(temp_filename)
            clips_audio.append(audio_clip)
            duracion = audio_clip.duration
            
            text_img = create_text_image(segmento)
            txt_clip = (ImageClip(text_img, transparent=True)  #Hace el fondo transparente
                      .set_start(tiempo_acumulado)
                      .set_duration(duracion)
                      .set_position('center'))
            
            video_segment = txt_clip.set_audio(audio_clip.set_start(tiempo_acumulado))
            clips_finales.append(video_segment)
            
            tiempo_acumulado += duracion
            time.sleep(0.2)

        # Calcular la duración total del video
        video_duration = tiempo_acumulado

        # Calcular cuántas veces necesitamos repetir el clip de fondo
        num_loops = int(video_duration / background_clip.duration) + 1

        # Crear una lista de clips de fondo repetidos
        background_clips = [background_clip] * num_loops

        # Concatenar los clips de fondo
        background_clip_repeated = concatenate_videoclips(background_clips, method="compose")

        # Ajustar el clip de fondo para que coincida con la duración total del video
        background_clip = background_clip_repeated.subclip(0, video_duration)

        # Superponer los clips de texto sobre el fondo
        final_video_with_background = CompositeVideoClip([background_clip] + clips_finales)

        # Generar el video en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            temp_video_path = temp_video_file.name
            logging.info(f"Escribiendo video a archivo temporal: {temp_video_path}")
            final_video_with_background.write_videofile(
                temp_video_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                preset='ultrafast',
                threads=4
            )
            logging.info("Video escrito exitosamente al archivo temporal.")

        success = True
        message = "Video generado exitosamente"


    except Exception as e:
        message = str(e)
        logging.error(f"Error: {message}")
        success = False

    finally: # Asegura que los recursos se liberen incluso si hay errores
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

        try:
            background_clip.close()
            background_clip_repeated.close()
        except:
            pass
            
        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.close(os.open(temp_file, os.O_RDONLY))
                    os.remove(temp_file)
            except:
                pass
        
        return success, message, temp_video_path

def main():
    st.title("Creador de Videos Automático")
    
    uploaded_file = st.file_uploader("Carga un archivo de texto", type="txt")
    voz_seleccionada = st.selectbox("Selecciona la voz", options=list(VOCES_DISPONIBLES.keys()))
    background_video = st.file_uploader("Carga un video de fondo (MP4)", type=["mp4"])
    
    if uploaded_file and background_video:
        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")
        
        # Guardar el video de fondo temporalmente
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                background_video_path = os.path.join(temp_dir, "background.mp4")
                with open(background_video_path, "wb") as f:
                    f.write(background_video.read())
                
                if st.button("Generar Video"):
                    with st.spinner('Generando video...'):
                        success, message, temp_video_path = create_simple_video(texto, nombre_salida, voz_seleccionada, background_video_path)
                        if success:
                            st.success(message)
                            try:
                                with open(temp_video_path, 'rb') as file:
                                    st.video(file) # Muestra el video directamente desde el archivo temporal
                                    st.download_button(label="Descargar video", data=file, file_name=f"{nombre_salida}.mp4")
                            except Exception as e:
                                st.error(f"Error al mostrar/descargar el video: {e}")
                            finally:
                                if temp_video_path:
                                    os.remove(temp_video_path) # Elimina el archivo temporal después de mostrarlo/descargarlo
                        else:
                            st.error(f"Error al generar video: {message}")

        except Exception as e:
            st.error(f"Error al procesar el video: {e}")

if __name__ == "__main__":
    main()
