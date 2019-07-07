import os

from PyQt5.QtCore import QTimer

import util
# import train_model
from trainer import training
from recognizer import load_gmm_model, recognize_file
import voice_util
import numpy as np
import threading
import time
import shutil
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QGraphicsScene, QGraphicsView, QLabel, QLineEdit, QPushButton, \
    QWidget, QHBoxLayout
from screeninfo import get_monitors
from voice_util import reduce_noise_power


def show_message(msg_type, msg):
    QtWidgets.QMessageBox.information(None, msg_type, msg)


def get_screen_resolution():
    # first monitor
    monitor = get_monitors()[0]
    return monitor.width, monitor.height


class Ui_Dialog(object):
    timer: QTimer
    horizontalLayout: QHBoxLayout
    layoutWidget_timer: QWidget
    label_id: QLabel
    label_name: QLabel
    msg_voice_existence: QLabel
    msg_question: QLabel
    edit_id: QLineEdit
    edit_name: QLineEdit
    btnEnter: QPushButton
    btnExit: QPushButton
    btn_record_true: QPushButton

    def __init__(self):
        self.isDataLoad = False
        self.isRecording = False
        self.recording_time = 30
        self.remain_recording_time = 30
        self.recording_file_path = ""
        self.data_path = "Database"
        self.training_path = "voice"
        self.model_path = os.path.join("model")
        self.ref_path = os.path.join("model", "DB.csv")
        self.model = None
        self.trained_voice_list = []
        self.database_list = {}
        self.ref_list = {}
        self.record_thread = None

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        width, height = get_screen_resolution()
        Dialog.resize(width, height)
        Dialog.setStyleSheet("background-color: #222;\n"
                            "color: #CCC;\n"
                            "font: 18pt \"MS Shell Dlg 2\";")

        self.label_id = QtWidgets.QLabel(Dialog)
        self.label_id.setGeometry(QtCore.QRect(int(width//4), 50, 100, 50))
        self.label_id.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_id.setObjectName("label_id")

        self.edit_id = QtWidgets.QLineEdit(Dialog)
        self.edit_id.setGeometry(QtCore.QRect(int(width//4)+150, 50, int(width//4), 50))
        self.edit_id.setObjectName("edit_id")

        self.btnEnter = QtWidgets.QPushButton(Dialog)
        self.btnEnter.setGeometry(QtCore.QRect(int(width//2)+200, 50, 200, 50))
        self.btnEnter.setObjectName("btnEnter")

        self.label_name = QtWidgets.QLabel(Dialog)
        self.label_name.setGeometry(QtCore.QRect(int(width//4), 150, 100, 50))
        self.label_name.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_name.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_name.setObjectName("label_name")

        self.edit_name = QtWidgets.QLineEdit(Dialog)
        self.edit_name.setGeometry(QtCore.QRect(int(width//4)+150, 150, int(width//4), 50))
        self.edit_name.setReadOnly(True)
        self.edit_name.setObjectName("edit_name")

        self.msg_voice_existence = QtWidgets.QLabel(Dialog)
        self.msg_voice_existence.setGeometry(QtCore.QRect(
            50, int(height//4), int(width//2)-50, 100))
        self.msg_voice_existence.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.msg_voice_existence.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.msg_voice_existence.setObjectName("msg_voice_existence")

        self.msg_question = QtWidgets.QLabel(Dialog)
        self.msg_question.setGeometry(QtCore.QRect(
            50, int(height // 3)+100, int(width // 2) - 50, 100))
        self.msg_question.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.msg_question.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.msg_question.setObjectName("msg_question")

        self.btn_record_true = QtWidgets.QPushButton(Dialog)
        self.btn_record_true.setGeometry(QtCore.QRect(
            int(width // 4)-120, int(height // 3)+250, 100, 50))
        self.btn_record_true.setObjectName("btn_true_record")

        self.btn_record_false = QtWidgets.QPushButton(Dialog)
        self.btn_record_false.setGeometry(QtCore.QRect(
            int(width // 4)+20, int(height // 3)+250, 100, 50))
        self.btn_record_false.setObjectName("btn_false_record")

        self.btn_recording = QtWidgets.QPushButton(Dialog)
        self.btn_recording.setGeometry(QtCore.QRect(
            200, int(height // 3)+250, int(width // 2)-400, 50))
        self.btn_recording.setObjectName("btn_recording")

        self.btn_training = QtWidgets.QPushButton(Dialog)
        self.btn_training.setGeometry(QtCore.QRect(
            int(width//4)-150, 3*int(height // 4), 300, 50))
        self.btn_training.setObjectName("btn_training")

        self.btnExit = QtWidgets.QPushButton(Dialog)
        self.btnExit.setGeometry(QtCore.QRect(width-150, 50, 100, 50))
        self.btnExit.setObjectName("btnExit")

        self.label_spectrum = QtWidgets.QLabel(Dialog)
        self.label_spectrum.setGeometry(QtCore.QRect(
            3*int(width//4)-200, int(height//4)-70, 400, 50))
        self.label_spectrum.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_spectrum.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.label_spectrum.setObjectName("label_spectrum")

        self.voiceSpectr = QGraphicsView(Dialog)
        self.voiceSpectr.setGeometry(QtCore.QRect(
            int(width // 2)+50, int(height//4), int(width // 2)-100, int(height//2)))
        self.voiceSpectr.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.voiceSpectr.setObjectName("voiceSpectr")
        self.voiceSpectr.scene = QGraphicsScene()
        self.voiceSpectr.setScene(self.voiceSpectr.scene)
        self.voiceSpectr.scene.setBackgroundBrush(QtCore.Qt.black)
        self.voiceSpectr.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.voiceSpectr.setDragMode(QGraphicsView.RubberBandDrag)

        self.label_result = QtWidgets.QLabel(Dialog)
        self.label_result.setGeometry(QtCore.QRect(
            int(width // 2)+50, 3*int(height//4)+50, int(width // 2)-100, 100))
        self.label_result.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_result.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.label_result.setObjectName("label_result")

        self.layoutWidget_timer = QtWidgets.QWidget(Dialog)
        self.layoutWidget_timer.setGeometry(QtCore.QRect(50, 50, 100, 30))
        self.layoutWidget_timer.setObjectName("layoutWidget_timer")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget_timer)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # signal and slot
        self.btnExit.clicked.connect(Dialog.accept)
        self.btnEnter.clicked.connect(self.id_checking)
        self.btn_record_true.clicked.connect(self.show_recording_button)
        self.btn_record_false.clicked.connect(self.hide_recording_button)
        self.btn_recording.clicked.connect(self.record_start)
        self.btn_training.clicked.connect(self.model_training)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # Initialize
        self.initialize()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Voice Authentication with real-time speaking"))
        self.label_id.setText(_translate("Dialog", "ID."))
        self.btnEnter.setText(_translate("Dialog", "Enter"))
        self.btn_record_true.setText(_translate("Dialog", "YES"))
        self.btn_record_false.setText(_translate("Dialog", "NO"))
        self.btn_recording.setText(_translate("Dialog", "Real Time Speaking"))
        self.btn_training.setText(_translate("Dialog", "TRAIN MODEL"))
        self.label_name.setText(_translate("Dialog", "NAME."))
        self.label_spectrum.setText(_translate("Dialog", "Voice Spectrum"))
        self.btnExit.setText(_translate("Dialog", "Exit"))

        self.update_time()
        self.timer = QtCore.QTimer(self.layoutWidget_timer)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

    def update_time(self):
        current = QtCore.QDateTime.currentDateTime()
        if self.isRecording:
            self.remain_recording_time -= 1
            self.btn_recording.setText("Stop Recording ({}s)".format(self.remain_recording_time))
            if self.remain_recording_time == 0:
                self.record_stop()

    def visible_buttons(self, status):
        self.btn_record_true.setVisible(status)
        self.btn_record_false.setVisible(status)

    def load_database(self):
        self.database_list.clear()
        if not os.path.exists(self.data_path):
            return
        for voice_file in os.listdir(self.data_path):
            if os.path.isdir(os.path.join(self.data_path, voice_file)):
                continue
            if not voice_file.lower().endswith('.wav'):
                continue
            segs = voice_file[:-4].split("_")
            tmp = {}
            try:
                _name = segs[0]
                _ID = int(segs[1])
                _index = int(segs[2])
                tmp["name"] = _name
                tmp["ID"] = _ID
                tmp["index"] = _index
            except:
                continue
            if _ID not in self.database_list.keys():
                self.database_list[_ID] = [tmp]
            else:
                self.database_list[_ID].append(tmp)

    def initialize(self):
        self.btn_training.setVisible(False)
        self.btn_recording.setVisible(False)
        self.visible_buttons(False)
        self.ref_list.clear()
        with open(self.ref_path, "rt") as fp:
            for line in fp:
                _ID, _name, _exist = line.strip().split(",")
                if int(_ID) not in self.ref_list.keys():
                    self.ref_list[int(_ID)] = [_name, _exist]

        self.load_database()
        self.isDataLoad = True
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.msg_question.setText("Model does not exist.")
            return
        # self.model = util.load_speaker_model(self.model_path)
        self.model, spk_list = load_gmm_model(self.model_path)
        for voice_id in spk_list:
            self.trained_voice_list.append(voice_id)
        self.msg_question.setText("Successfully loaded.")

    def model_training(self):
        self.msg_question.setText("now training...")
        # train_model.train(3, self.model_path)
        training(self.training_path, self.model_path)
        self.load_model()

    def show_recording_button(self):
        self.visible_buttons(False)
        self.label_result.clear()
        self.btn_recording.setText("Real Time Speaking ({}s)".format(self.recording_time))
        self.btn_recording.setVisible(True)

    def hide_recording_button(self):
        self.msg_question.clear()
        self.visible_buttons(False)
        self.btn_recording.setVisible(False)

    def check_status(self):
        name_id = self.edit_id.text()
        if not name_id.isnumeric():
            return "None", -1
        _ID = int(name_id)
        if _ID not in self.ref_list.keys():
            return "None", -1
        _status = self.ref_list[_ID][1]
        return _status, _ID

    def id_checking(self):
        self.edit_name.clear()
        self.msg_voice_existence.clear()
        self.msg_question.clear()
        self.label_result.clear()
        self.voiceSpectr.scene.clear()
        self.visible_buttons(False)
        self.btn_recording.setVisible(False)
        
        _id = self.edit_id.text()
        if not _id.isnumeric():
            self.msg_question.setText("Please input voice ID as integer")
            show_message("Error!", "Please input voice ID as integer")
            return
        _ID = int(_id)
        if _ID not in self.ref_list.keys():
            self.msg_question.setText("This ID does not exist in database")
            show_message("Error!", "This ID does not exist in database")
            return
        name = self.ref_list[_ID][0]
        self.edit_name.setText(name)
        if self.ref_list[_ID][1] == "PRESENT":
            self.msg_voice_existence.setText("Voice Sample Present")
            self.msg_question.setText("How can I help you?")
            self.btn_recording.setText("Real Time Speaking")
            self.btn_recording.setVisible(True)
            self.recording_time = 30

            # show voice over
            first_file = self.database_list[_ID][0]
            file_name = "{}_{}_{}.wav".format(first_file["name"], first_file["ID"], first_file["index"])
            file_path = os.path.join(self.data_path, file_name)
            data = voice_util.read_voice_file(file_path)
            self.draw_audio_data(data)
            self.play_voice_file(file_path)

            # self.make_training_data()
            # self.model_training()

        else:
            self.msg_voice_existence.setText("Voice Sample Absent")
            self.msg_question.setText("Do you wish to record?")
            self.visible_buttons(True)
            self.recording_time = 60

    def make_training_data_other(self):
        if not self.isDataLoad:
            return

        # remove previous all
        if os.path.exists(self.training_path):
            if voice_util.get_platform() == 'win32':
                os.system("rmdir {} /s /q".format(self.training_path))
            else:
                os.system('rm -rf {}'.format(self.training_path))
        os.mkdir(self.training_path)

        # copy samples into "voice" folder
        _ID = self.get_id()
        name = self.database_list[int(_ID)][0]['name']

        dst_path1 = os.path.join(self.training_path, _ID)
        os.mkdir(dst_path1)

        dst_path2 = os.path.join(self.training_path, "Other")
        os.mkdir(dst_path2)

        # copy from _name
        for ele in self.database_list[int(_ID)]:
            file_name = "{}_{}_{}.wav".format(ele["name"], ele["ID"], ele["index"])
            src_file_path = os.path.join(self.data_path, file_name)
            dst_file_path = os.path.join(dst_path1, file_name)
            try:
                shutil.copy(src_file_path, dst_file_path)
            except:
                continue

        # copy from other samples
        other_voice_counts = len(self.database_list) - 1
        each_sample_counts = int(10 // other_voice_counts)
        for new_ID in self.database_list.keys():
            if new_ID == int(_ID):
                continue
            cnt = 0
            for ele in self.database_list[new_ID]:
                if cnt >= each_sample_counts:
                    break
                file_name = "{}_{}_{}.wav".format(ele["name"], ele["ID"], ele["index"])
                src_file_path = os.path.join(self.data_path, file_name)
                dst_file_path = os.path.join(dst_path2, file_name)
                try:
                    shutil.copy(src_file_path, dst_file_path)
                    cnt += 1
                except:
                    continue

    def make_training_data(self):
        if not self.isDataLoad:
            return

        if os.path.exists(self.training_path):
            if voice_util.get_platform() == 'win32':
                os.system("rmdir {} /s /q".format(self.training_path))
            else:
                os.system('rm -rf {}'.format(self.training_path))
        os.mkdir(self.training_path)

        # copy samples into "voice" folder
        for _ID in self.database_list.keys():
            dst_path = os.path.join(self.training_path, "{}".format(_ID))
            os.mkdir(dst_path)
            for ele in self.database_list[_ID]:
                file_name = "{}_{}_{}.wav".format(ele["name"], ele["ID"], ele["index"])
                src_file_path = os.path.join(self.data_path, file_name)
                dst_file_path = os.path.join(dst_path, file_name)
                try:
                    shutil.copy(src_file_path, dst_file_path)
                except:
                    continue

    def get_id(self):
        str_id = self.edit_id.text()
        if not str_id.isnumeric():
            show_message("Error!", "Invalid voice ID format!")
            return 0
        _ID = int(str_id)
        return str_id

    def record_start(self):
        if not self.isRecording:
            self.isRecording = True
            self.remain_recording_time = self.recording_time
            self.btn_recording.setText("Stop Recording ({}s)"
                                      .format(self.remain_recording_time))
            voice_util.name_id = self.get_id()
            self.record_thread = threading.Thread(target=voice_util.record_from_mic)
            self.record_thread.start()
        else:
            self.record_stop()

            voice_util.record_stop()
            time.sleep(1)
            if voice_util.rec_filename:
                self.recording_file_path = voice_util.rec_filename
                data = voice_util.read_voice_file(self.recording_file_path)
                self.draw_audio_data(data)
                self.play_voice_file(self.recording_file_path)

                # in case of making of sample
                _cur_status, _ID = self.check_status()
                if _cur_status == "ABSENT":
                    _name = self.ref_list[_ID][0]
                    if voice_util.split_sample(self.recording_file_path, _ID, _name):
                        self.ref_list[_ID][1] = "PRESENT"
                        # update excel file
                        with open(self.ref_path, "wt") as fp:
                            for _ID in self.ref_list.keys():
                                _name = self.ref_list[_ID][0]
                                _exist = self.ref_list[_ID][1]
                                fp.write("{},{},{}\n".format(_ID, _name, _exist))
                else:
                    self.authentic_voice()
                    return
            self.load_database()
            self.make_training_data()

    def record_stop(self):
        self.isRecording = False
        self.remain_recording_time = self.recording_time
        self.btn_recording.setText("Real Time Speaking ({}s)".format(self.recording_time))

    def draw_audio_data(self, data):
        view_rc = self.voiceSpectr.size()
        zoom_rate = int(len(data) / view_rc.width())
        zoom_data = []
        for i in range(len(data) // zoom_rate - 1):
            zoom_data.append(np.mean(data[zoom_rate * i:zoom_rate * (i + 1)]))
        data = np.array(zoom_data)
        zoom_data.clear()

        data = data / np.max(np.abs(data))
        self.voiceSpectr.scene.clear()

        scene_view = self.voiceSpectr.scene.sceneRect()
        sig_amp = view_rc.height() // 2
        signal = sig_amp + sig_amp * data
        pen = QtGui.QPen(QtCore.Qt.green)

        for i in range(len(signal) - 1):
            x1 = signal[i]
            x2 = signal[i + 1]
            r = QtCore.QLineF(QtCore.QPointF(i, x1), QtCore.QPointF(i + 1, x2))
            self.voiceSpectr.scene.addLine(r, pen)

    def play_voice_file(self, filename):
        voice_util.play_wave(filename, True)

    def authentic_voice(self):
        if not os.path.exists(self.recording_file_path):
            return
        if not self.model:
            show_message("Info!", "Model does not loaded.")
            return
        # get ID and name
        _ID = self.get_id()

        voice, score = recognize_file(self.model,
                                self.trained_voice_list,
                                self.recording_file_path
                                )
        # voice, score = util.recognize(self.model, self.recording_file_path)

        if _ID == voice:
            name = self.database_list[int(voice)][0]
            name = name['name']
            if score > 0.8:
                self.label_result.setText("Verification Result:  AUTHENTIC ({}) ({:.2f})".format(name, score))
            else:
                self.label_result.setText("Verification Result:  NOT AUTHENTIC ({:.2f})".format(score))
        else:
            self.label_result.setText("Verification Result:  NOT AUTHENTIC ({:.2f})".format(score))
