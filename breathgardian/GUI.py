# === Imports ===
import sys
import os
import re

# PyQt5 UI components
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QFileDialog, QSplitter, QLabel, QStackedWidget
)
from PyQt5.QtCore import Qt

# 3D visualization and plotting
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Backend processing
from src.load_tensor import load_3d_image_from_folder
from src.Show2D import show2D
from src.Superpose2D import show2D_overlay_on_canvas
from src.Show3DData import show3D_ct_mask_overlay
from src.find_peak import find_second_bump_limits
from src.predict import predict, adjust_depth


# === Main Application Window ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breath Guardian")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        self.init_ui()

    # === UI Initialization ===
    def init_ui(self):
        central_widget = QWidget()
        central_layout = QVBoxLayout()

        # Top buttons layout
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.param_button = QPushButton("Show 2D")
        self.param_button.clicked.connect(self.show_2d_handler)

        self.vis_button = QPushButton("Show 3D")
        self.vis_button.clicked.connect(self.show3d_handler)

        self.overlay_button = QPushButton("Overlay")
        self.overlay_button.clicked.connect(self.overlay_handler)

        self.cluster_button = QPushButton("Predict")
        self.cluster_button.clicked.connect(self.predict_handler)

        # Add styled buttons to top layout
        for button in [self.param_button, self.vis_button, self.cluster_button, self.overlay_button]:
            button.setStyleSheet(self.button_style())
            top_layout.addWidget(button)

        top_layout.addStretch()

        # Main content splitter: sidebar + content area
        main_splitter = QSplitter(Qt.Horizontal)

        # === Sidebar: folder + file list ===
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()

        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.setStyleSheet(self.open_folder_button_style())
        self.open_folder_button.clicked.connect(self.open_folder)

        self.file_list_widget = QListWidget()
        self.file_list_widget.setStyleSheet(self.list_widget_style())

        sidebar_layout.addWidget(self.open_folder_button)
        sidebar_layout.addWidget(self.file_list_widget)
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setMaximumWidth(250)

        # === Right Widget: contains stacked 2D and 3D views ===
        self.right_widget = QStackedWidget()

        # 2D View setup
        self.view_2d_widget = QWidget()
        view_2d_layout = QHBoxLayout()
        self.view_2d_widget.setLayout(view_2d_layout)

        self.right_splitter = QSplitter(Qt.Horizontal)
        self.canvas_2d, self.figure_2d = self.create_matplotlib_widget()
        self.canvas_pred, self.figure_pred = self.create_matplotlib_widget()
        self.canvas_overlay, self.figure_overlay = self.create_matplotlib_widget()
        self.canvas_overlay.hide()

        self.right_splitter.addWidget(self.canvas_2d)
        self.right_splitter.addWidget(self.canvas_pred)
        self.right_splitter.addWidget(self.canvas_overlay)
        self.right_splitter.setSizes([800, 800, 0])

        view_2d_layout.addWidget(self.right_splitter)

        # 3D View setup
        self.view_3d_widget = QWidget()
        view_3d_layout = QVBoxLayout()
        self.view_3d_widget.setLayout(view_3d_layout)

        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setStyleSheet("background-color: #000000; border: 1px solid #0D47A1;")
        view_3d_layout.addWidget(self.vtk_widget)

        # Add both views to stacked widget
        self.right_widget.addWidget(self.view_2d_widget)
        self.right_widget.addWidget(self.view_3d_widget)
        self.right_widget.setCurrentIndex(0)  # Start in 2D view

        # Finalize layout
        main_splitter.addWidget(sidebar_widget)
        main_splitter.addWidget(self.right_widget)
        main_splitter.setSizes([300, 1300])

        central_layout.addLayout(top_layout)
        central_layout.addWidget(main_splitter)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # VTK interactor setup
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

    # === Folder Selection ===
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = os.listdir(folder)
            pattern = r"^data-\d+-\d+\.npy$"
            npyfiles = [f for f in files if re.match(pattern, f)]

            if npyfiles:
                self.selected_folder = folder
                self.file_list_widget.clear()
                npyfiles.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
                self.file_list_widget.addItems(npyfiles)

                self.data_tensor = load_3d_image_from_folder(self.selected_folder)
                self.mask_tensor = predict(self.data_tensor, 0.8)
                self.data_tensor = adjust_depth(self.data_tensor, target_depth=288)

    # === Matplotlib canvas creation ===
    def create_matplotlib_widget(self):
        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #000000; border: 1px solid #0D47A1;")
        return canvas, fig

    # === Button Handlers ===
    def show_2d_handler(self):
        if hasattr(self, 'selected_folder'):
            self.state = 2
            show2D(self, self.data_tensor, self.figure_2d, self.canvas_2d)
            self.right_widget.setCurrentIndex(0)
        else:
            print("No folder selected yet!")

    def show3d_handler(self):
        self.state = 3
        if hasattr(self, 'data_tensor'):
            self.right_widget.setCurrentIndex(1)
            self.vtk_widget.GetRenderWindow().GetRenderers().RemoveAllItems()

            try:
                limits = find_second_bump_limits(self.data_tensor, bins=100, plot=False, threshold=0.05)
                wl = (limits[0] + limits[1]) / 2
            except:
                wl = 0.26

            print(wl, type(wl))
            show3D_ct_mask_overlay(self.data_tensor, None, self.vtk_widget, wl)

            if not self.vtk_interactor.GetInitialized():
                self.vtk_interactor.Initialize()
            self.vtk_widget.GetRenderWindow().Render()
        else:
            print("No CT volume loaded.")

    def predict_handler(self):
        if hasattr(self, 'data_tensor') and hasattr(self, 'mask_tensor'):
            if self.state == 2:
                show2D(self, self.mask_tensor, self.figure_pred, self.canvas_pred)
                self.right_widget.setCurrentIndex(0)
            elif self.state == 3:
                self.right_widget.setCurrentIndex(1)
                self.vtk_widget.GetRenderWindow().GetRenderers().RemoveAllItems()
                show3D_ct_mask_overlay(None, self.mask_tensor, self.vtk_widget)
                if not self.vtk_interactor.GetInitialized():
                    self.vtk_interactor.Initialize()
                self.vtk_widget.GetRenderWindow().Render()
        else:
            print("No folder selected yet or no mask/data loaded.")

    def overlay_handler(self):
        if hasattr(self, 'data_tensor') and hasattr(self, 'mask_tensor'):
            if self.state == 2:
                self.right_widget.setCurrentIndex(0)
                self.canvas_2d.hide()
                self.canvas_pred.hide()
                self.canvas_overlay.show()
                self.figure_overlay.clear()
                show2D_overlay_on_canvas(self.data_tensor, self.mask_tensor, self.figure_overlay, self.canvas_overlay)
            elif self.state == 3:
                try:
                    limits = find_second_bump_limits(self.data_tensor, bins=100, plot=False, threshold=0.05)
                    wl = (limits[0] + limits[1]) / 2
                except:
                    wl = 0.26
                print(wl, type(wl))
                show3D_ct_mask_overlay(self.data_tensor, self.mask_tensor, self.vtk_widget, wl)
        else:
            print("Missing CT or mask data.")

    # === Stylesheets ===
    def button_style(self):
        return """
            QPushButton {
                background-color: #0D47A1;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """

    def open_folder_button_style(self):
        return """
            QPushButton {
                background-color: #1E88E5;
                color: white;
                padding: 6px 12px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
        """

    def list_widget_style(self):
        return """
            QListWidget {
                background-color: #1A1A1A;
                color: white;
                border: 1px solid #0D47A1;
                padding: 5px;
            }
        """


# === Application Entry Point ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
