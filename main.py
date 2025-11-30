import cv2
import time
import face_recognition
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import os
import traceback
import streamlit as st
from PIL import Image
import tempfile
import base64
from datetime import datetime, timedelta
import pickle
import json

class FaceRecognitionApp:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.running = True
        self.model_dir = "saved_models"
        self.auto_save_enabled = True
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.load_saved_model()
    
    def get_model_path(self, filename):
        """Mendapatkan path lengkap untuk file model"""
        return os.path.join(self.model_dir, filename)
    
    def save_model(self):
        """Menyimpan model encoding wajah ke file"""
        try:
            if not self.auto_save_enabled:
                return False
                
            model_data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            
            with open(self.get_model_path('face_encodings.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            metadata = {
                'names': self.known_face_names,
                'total_faces': len(self.known_face_names),
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.get_model_path('face_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model berhasil disimpan: {len(self.known_face_names)} wajah")
            return True
            
        except Exception as e:
            print(f"Error menyimpan model: {e}")
            return False
    
    def load_saved_model(self):
        """Memuat model encoding wajah dari file"""
        try:
            model_path = self.get_model_path('face_encodings.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.known_face_encodings = model_data['encodings']
                self.known_face_names = model_data['names']
                
                print(f"Model berhasil dimuat: {len(self.known_face_names)} wajah")
                return True
            else:
                print("Tidak ada model yang tersimpan")
                return False
                
        except Exception as e:
            print(f"Error memuat model: {e}")
            return False
    
    def auto_save(self):
        """Fungsi auto save yang dipanggil setelah perubahan data wajah"""
        if self.auto_save_enabled:
            return self.save_model()
        return False

    def load_known_faces(self, image_path, name):
        """Load wajah yang sudah dikenal dengan nama"""
        try:
            if not os.path.exists(image_path):
                st.warning(f"PERINGATAN: File {image_path} tidak ditemukan!")
                return False
                
            st.info(f"Memuat foto {image_path} untuk {name}...")
            wajah_image = face_recognition.load_image_file(image_path)
            encoding_list = face_recognition.face_encodings(wajah_image)
            
            if not encoding_list:
                st.error(f"PERINGATAN: Tidak ada wajah yang terdeteksi di {image_path}!")
                return False
                
            self.known_face_encodings.append(encoding_list[0])
            self.known_face_names.append(name)
            
            if self.auto_save():
                st.success(f"✓ Berhasil memuat data wajah untuk {name} (Auto-saved)")
            else:
                st.success(f"✓ Berhasil memuat data wajah untuk {name}")
                
            return True
            
        except Exception as e:
            st.error(f"Error saat memuat wajah: {e}")
            return False

    def delete_face(self, name):
        """Menghapus wajah dari daftar known faces"""
        try:
            if name in self.known_face_names:
                idx = self.known_face_names.index(name)
                self.known_face_names.pop(idx)
                self.known_face_encodings.pop(idx)
                
                if self.auto_save():
                    st.success(f"Wajah {name} berhasil dihapus (Auto-saved)")
                else:
                    st.success(f"Wajah {name} berhasil dihapus")
                    
                return True
            return False
        except Exception as e:
            st.error(f"Error menghapus wajah: {e}")
            return False

    def process_image(self, image):
        """Process single image for face recognition"""
        try:
            detector = FaceDetector()
            
            if isinstance(image, Image.Image):
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                frame = image.copy()
            
            frame, faces = detector.findFaces(frame, draw=True)
            recognized_faces = []
            
            if faces and self.known_face_encodings:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Tidak Dikenal"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                    
                    recognized_faces.append(name)
                    
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, bottom + 5), (right, bottom + 30), (0, 0, 0), -1)
                    color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
                    cv2.putText(frame, name, (left + 5, bottom + 22), self.font, 0.6, color, 2)
            
            result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return result_image, len(faces), recognized_faces
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, 0, []

    def get_available_cameras(self):
        """Mendapatkan daftar kamera yang tersedia"""
        available_cameras = []
        
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
            else:
                cap.release()
                
        return available_cameras

def main_streamlit():
    """Streamlit application"""
    st.set_page_config(
        page_title="Face Recognition App",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Recognition Application")
    st.markdown("Aplikasi pengenalan wajah dengan Streamlit - Multi Wajah & Real-time Streaming")
    
    if 'app' not in st.session_state:
        st.session_state.app = FaceRecognitionApp()
        st.session_state.known_faces_loaded = False
        st.session_state.webcam_active = False
        st.session_state.face_recognized = False
        st.session_state.last_capture_time = None
        st.session_state.recognized_name = None
        st.session_state.available_cameras = []
        st.session_state.selected_camera = 0
        st.session_state.known_faces_list = st.session_state.app.known_face_names.copy()
        st.session_state.operation_mode = "Streaming Real-time" 
    
    app = st.session_state.app
    
    st.sidebar.header("🔧 Konfigurasi Wajah")
    
    with st.sidebar.expander("➕ Tambah Wajah Baru", expanded=True):
        face_name = st.text_input("Nama Wajah", placeholder="Masukkan nama", key="face_name")
        
        uploaded_file = st.file_uploader(
            "Upload foto wajah", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload foto wajah yang jelas",
            key="reference_upload"
        )
        
        if st.button("✅ Tambah Wajah", use_container_width=True) and uploaded_file is not None and face_name:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if app.load_known_faces(tmp_path, face_name):
                st.session_state.known_faces_loaded = True
                st.session_state.known_faces_list = app.known_face_names.copy()
                st.rerun()
            
            os.unlink(tmp_path)
    
    if st.session_state.known_faces_list:
        with st.sidebar.expander(f"👥 Wajah Terdaftar ({len(st.session_state.known_faces_list)})", expanded=True):
            for i, name in enumerate(st.session_state.known_faces_list):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        if app.delete_face(name):
                            st.session_state.known_faces_list = app.known_face_names.copy()
                            st.rerun()
    else:
        with st.sidebar.expander("👥 Wajah Terdaftar", expanded=True):
            st.info("Belum ada wajah yang terdaftar")
    
    with st.sidebar.expander("💾 Auto Save Settings", expanded=True):
        st.info("Model otomatis disimpan saat menambah/menghapus wajah")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Model Manual", use_container_width=True):
                if app.save_model():
                    st.success("Model berhasil disimpan!")
                else:
                    st.error("Gagal menyimpan model")
        
        with col2:
            if st.button("🔄 Load Model", use_container_width=True):
                if app.load_saved_model():
                    st.session_state.known_faces_list = app.known_face_names.copy()
                    st.success("Model berhasil dimuat!")
                    st.rerun()
                else:
                    st.error("Gagal memuat model")
        
        if st.session_state.known_faces_list:
            st.metric("Total Wajah Tersimpan", len(st.session_state.known_faces_list))

    with st.sidebar.expander("🎥 Konfigurasi Webcam", expanded=True):
        if not st.session_state.available_cameras:
            with st.spinner("Mendeteksi kamera..."):
                st.session_state.available_cameras = app.get_available_cameras()
        
        if st.session_state.available_cameras:
            st.session_state.selected_camera = st.selectbox(
                "Pilih Kamera:",
                options=st.session_state.available_cameras,
                index=0,
                format_func=lambda x: f"Kamera {x} {'(Default)' if x == 0 else ''}"
            )
            st.success(f"✅ {len(st.session_state.available_cameras)} kamera terdeteksi")
        else:
            st.error("❌ Tidak ada kamera yang terdeteksi!")
        
        st.session_state.operation_mode = st.radio(
            "Mode Operasi:",
            ["Streaming Real-time", "Auto Capture & Stop"],
            help="Streaming: Tampilkan video terus menerus. Auto Stop: Berhenti otomatis ketika wajah dikenali"
        )
    
    tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "🎥 Webcam Real-time", "ℹ️ Informasi"])
    
    with tab1:
        st.header("Deteksi Wajah dari Gambar")
        
        uploaded_image = st.file_uploader(
            "Upload gambar untuk deteksi wajah",
            type=['jpg', 'jpeg', 'png'],
            key="image_upload"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Gambar Asli", width='stretch')
            
            with col2:
                with st.spinner("Memproses gambar..."):
                    result_image, face_count, recognized_faces = app.process_image(image)
                    
                    if result_image is not None:
                        st.image(result_image, caption=f"Hasil Deteksi ({face_count} wajah ditemukan)", width='stretch')
                        
                        if face_count > 0:
                            if st.session_state.known_faces_loaded:
                                if recognized_faces and any(name != "Tidak Dikenal" for name in recognized_faces):
                                    recognized_names = [name for name in recognized_faces if name != "Tidak Dikenal"]
                                    st.success(f"✅ Ditemukan {face_count} wajah!")
                                    st.success(f"👤 Wajah dikenali: {', '.join(set(recognized_names))}")
                                else:
                                    st.success(f"✅ Ditemukan {face_count} wajah!")
                                    st.warning("❌ Tidak ada wajah yang dikenali")
                            else:
                                st.warning(f"✅ Ditemukan {face_count} wajah (tanpa pengenalan - tambahkan wajah referensi dulu)")
                        else:
                            st.warning("❌ Tidak ada wajah yang terdeteksi")
    
    with tab2:
        st.header("Webcam Real-time Streaming")
        
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'face_recognized' not in st.session_state:
            st.session_state.face_recognized = False
        if 'recognized_name' not in st.session_state:
            st.session_state.recognized_name = None
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if not st.session_state.webcam_active:
                if st.session_state.available_cameras:
                    if st.button("🎬 Mulai Webcam", type="primary", use_container_width=True):
                        st.session_state.webcam_active = True
                        st.session_state.face_recognized = False
                        st.session_state.recognized_name = None
                        st.session_state.last_capture_time = datetime.now()
                        st.rerun()
                else:
                    st.button("🎬 Mulai Webcam", type="primary", use_container_width=True, disabled=True)
                    st.error("Tidak ada kamera yang tersedia!")
            else:
                if st.button("⏹️ Stop Webcam", type="secondary", use_container_width=True):
                    st.session_state.webcam_active = False
                    st.rerun()
            
            if st.session_state.webcam_active:
                st.info(f"🔴 Webcam aktif (Kamera {st.session_state.selected_camera})")
                st.info(f"⏱️ Mode: {st.session_state.operation_mode}")
                
                if st.session_state.face_recognized:
                    st.success(f"✅ Wajah dikenali: {st.session_state.recognized_name}")
                    st.balloons()
        
        with col2:
            webcam_placeholder = st.empty()
            status_placeholder = st.empty()

            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(st.session_state.selected_camera)
                
                if not cap.isOpened():
                    st.error(f"Tidak dapat mengakses kamera {st.session_state.selected_camera}!")
                    st.session_state.webcam_active = False
                else:
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)  
                        
                        detector = FaceDetector()
                        
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Gagal membaca frame dari kamera")
                                break
                
                            processed_frame, faces = detector.findFaces(frame, draw=True)
                            face_detected = False
                            
                            if faces and st.session_state.known_faces_loaded and app.known_face_encodings:
                                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                face_locations = face_recognition.face_locations(rgb_frame)
                                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                
                                for face_encoding, face_location in zip(face_encodings, face_locations):
                                    matches = face_recognition.compare_faces(app.known_face_encodings, face_encoding, tolerance=0.6)
                                    name = "Tidak Dikenal"
                                    
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        name = app.known_face_names[first_match_index]
                                        face_detected = True
                                        
                                        if st.session_state.operation_mode == "Auto Capture & Stop":
                                            st.session_state.face_recognized = True
                                            st.session_state.recognized_name = name
                                    
                                    top, right, bottom, left = face_location
                                    cv2.rectangle(processed_frame, (left, bottom + 5), (right, bottom + 30), (0, 0, 0), -1)
                                    color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
                                    cv2.putText(processed_frame, name, (left + 5, bottom + 22), app.font, 0.6, color, 2)
                            
                            cv2.putText(processed_frame, f'Wajah: {len(faces) if faces else 0}', (10, 30), app.font, 0.7, (255, 255, 0), 2)
                            cv2.putText(processed_frame, f'Kamera: {st.session_state.selected_camera}', (10, 60), app.font, 0.5, (255, 255, 255), 1)
                            cv2.putText(processed_frame, st.session_state.operation_mode, (10, 80), app.font, 0.5, (255, 255, 255), 1)
                            
                            if st.session_state.operation_mode == "Auto Capture & Stop" and st.session_state.face_recognized:
                                cv2.putText(processed_frame, "WAJAH DIKENALI!", (10, 110), app.font, 1, (0, 255, 0), 2)
                                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                webcam_placeholder.image(display_frame, caption="Webcam Live - Wajah Dikenali!", width='stretch')
                                break

                            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            webcam_placeholder.image(display_frame, caption="Webcam Live", width='stretch')
                            
                            time.sleep(0.03)  
                            
                    except Exception as e:
                        st.error(f"Error dalam webcam: {e}")
                        traceback.print_exc()
                    finally:
                        cap.release()
                
                if st.session_state.face_recognized and st.session_state.operation_mode == "Auto Capture & Stop":
                    st.success(f"✅ Wajah berhasil dikenali: {st.session_state.recognized_name}")
                    st.info("Webcam otomatis berhenti karena wajah sudah dikenali")
            
            else:
                if st.session_state.available_cameras:
                    webcam_placeholder.info("Klik 'Mulai Webcam' untuk memulai streaming")
                else:
                    webcam_placeholder.error("Tidak ada kamera yang terdeteksi. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
    
    with tab3:
        st.header("📋 Informasi Aplikasi")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Cara Menggunakan:")
            st.markdown("""
            1. **Tambah Wajah Referensi**: 
               - Masukkan nama di sidebar
               - Upload foto wajah yang jelas
               - Klik 'Tambah Wajah'
            
            2. **Deteksi dari Gambar**: 
               - Upload gambar di tab 'Upload Gambar'
               - Otomatis diproses dan menampilkan hasil
            
            3. **Webcam Real-time**: 
               - Pilih kamera di sidebar
               - Pilih mode operasi
               - Klik 'Mulai Webcam'
            """)
            
            st.subheader("Fitur:")
            st.markdown("""
            - ✅ Multi-wajah recognition
            - ✅ Streaming webcam real-time (30 FPS)
            - ✅ Auto-stop ketika wajah dikenali
            - ✅ Upload gambar langsung proses
            - ✅ Hapus wajah dari daftar
            - ✅ **Auto Save Model** (Fitur Baru)
            """)
        
        with col_info2:
            st.subheader("Mode Webcam:")
            st.markdown("""
            - **🎥 Streaming Real-time**: 
              Tampilkan video terus menerus dengan deteksi real-time
            
            - **⏹️ Auto Capture & Stop**: 
              Berhenti otomatis ketika wajah berhasil dikenali
            """)
            
            st.subheader("Fitur Auto Save:")
            st.markdown("""
            - **💾 Auto Save**: Model otomatis disimpan saat:
              - Menambah wajah baru
              - Menghapus wajah
            - **🔄 Auto Load**: Model otomatis dimuat saat aplikasi dibuka
            - **📁 Manual Save/Load**: Tombol manual untuk backup
            """)
            
            st.subheader("Teknologi:")
            st.markdown("""
            - OpenCV
            - face_recognition
            - cvzone
            - Streamlit
            - Pickle (untuk penyimpanan model)
            """)
            
            st.subheader("Tips:")
            st.markdown("""
            - Gunakan foto referensi dengan wajah yang jelas
            - Pencahayaan yang baik meningkatkan akurasi
            - Pastikan tidak ada aplikasi lain yang menggunakan kamera
            - Untuk hasil terbaik, gunakan mode 'Auto Capture & Stop'
            - Data wajah otomatis tersimpan di folder 'saved_models'
            """)

def main():
    """Main function"""
    print("=== FACE RECOGNITION APPLICATION ===")
    print("Menjalankan aplikasi Streamlit...")
    print("Buka http://localhost:8501 di browser Anda")
    
    try:
        main_streamlit()
    except ImportError as e:
        st.error(f"Package missing: {e}")
        st.info("Install required packages dengan:")
        st.code("pip install streamlit opencv-python face-recognition cvzone pillow numpy")
    except Exception as e:
        st.error(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()