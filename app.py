import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMenuBar, QMenu, QAction, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from app_ui import Ui_MainWindow  # 导入生成的 UI 类
from PyQt5 import QtGui
from PyQt5.QtCore import *
import cv2
import os
from sam_hq.demo.sam_get_mask import mask

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         # 设置主窗口
#         self.setWindowTitle('SAM Desktop App')
#         self.setGeometry(100, 100, 800, 600)

#         # 创建菜单栏
#         self.create_menu()

#         # 创建主显示区域
#         self.image_label = QLabel()
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.setCentralWidget(self.image_label)

#     def create_menu(self):
#         menubar = self.menuBar()

#         # 文件菜单
#         file_menu = menubar.addMenu('File')
#         open_action = QAction('Open', self)
#         open_action.triggered.connect(self.open_image)
#         file_menu.addAction(open_action)

#         exit_action = QAction('Exit', self)
#         exit_action.triggered.connect(self.close)
#         file_menu.addAction(exit_action)

#         # 编辑菜单（可以添加更多功能）
#         edit_menu = menubar.addMenu('Edit')
#         segment_action = QAction('Segment Image', self)
#         segment_action.triggered.connect(self.segment_image)
#         edit_menu.addAction(segment_action)

#     def open_image(self):
#         options = QFileDialog.Options()
#         file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
#         if file_name:
#             self.display_image(file_name)
#         pass

#     def segment_image(self):
#         # 这里实现图像分割功能
#         pass

#     def display_image(self, image_path):
#         pixmap = QPixmap(image_path)
#         self.image_label.setPixmap(pixmap)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())

class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Load image
        self.file_button.clicked.connect(self.open_image)
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_image)

        # Save image
        self.save_button.clicked.connect(self.save_image)
        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_image)

        # Remove background
        self.mask_button.clicked.connect(self.get_mask)
        save_action = QAction('Mask', self)
        save_action.triggered.connect(self.get_mask)

        self.gen_mask = mask()
        self.file_name = None

    def open_image(self):
        options = QFileDialog.Options()
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if self.file_name:
            self.display_image(self.file_name)
        pass
    
    def display_image(self, image_path):
        # self.image = cv2.imread(image_path)
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        self.graphicsView.setScene(scene)
        # self.graphicsView.setFixedSize(pixmap.width(), pixmap.height())
        self.graphicsView.fitInView(pixmap_item, mode=1)

    def save_image(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    
        if save_path:
            # 从场景中提取 QPixmap 项
            scene = self.graphicsView.scene()
            if scene:
            #     # 创建一个新的 QPixmap，并指定场景的大小
            #     pixmap = QPixmap(self.graphicsView.viewport().size())

            #     # 将 QGraphicsView 的内容渲染到 QPixmap 上
            #     painter = QtGui.QPainter(pixmap)
            #     self.graphicsView.render(painter)
            #     painter.end()
                rect = scene().sceneRect()
                pixmap = QtGui.QImage(rect.height(),rect.width(),QtGui.QImage.Format_ARGB32_Premultiplied)
                painter = QtGui.QPainter(pixmap)
                rectf = QRectF(0,0,pixmap.rect().height(),pixmap.rect().width())
                self.scene().render(painter,rectf,rect)

                # 保存图像
                pixmap.save(save_path)

    def get_mask(self):
        try:
            masked_image = self.gen_mask.get_mask(self.file_name)
        except Exception as e:
            print(f"Error generating mask: {e}")
            return

        # Convert to QImage and display in QGraphicsView
        height, width, channel = masked_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(masked_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Display the masked image
        pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(pixmap_item, mode=1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
