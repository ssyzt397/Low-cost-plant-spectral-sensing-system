import sys
import os
import io
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow import keras

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QGroupBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


FEATURE_COUNT = 8
WAVELENGTHS = np.array([415, 445, 480, 515, 555, 590, 630, 680])
LABEL_MAP = {
    0: "Keep Observation",
    1: "Replenish Water",
    2: "Dying Plant"
}

def load_models():
    models = {}

    if os.path.exists("./models/SVM_model.pkl"):
        models["SVM"] = joblib.load("./models/SVM_model.pkl")

    if os.path.exists("./models/RF_model.pkl"):
        models["Random Forest"] = joblib.load("./models/RF_model.pkl")

    if os.path.exists("./models/KNN_model.pkl"):
        models["KNN"] = joblib.load("./models/KNN_model.pkl")

    if os.path.exists("./models/DT_model.pkl"):
        models["Decision Tree"] = joblib.load("./models/DT_model.pkl")

    if os.path.exists("./models/BPNN_model.keras"):
        models["BPNN"] = keras.models.load_model("./models/BPNN_model.keras")

    if os.path.exists("./models/CNN_model.keras"):
        models["CNN"] = keras.models.load_model("./models/CNN_model.keras")

    return models


def ensemble_predict(models, features):
    x_input = np.array(features, dtype=float).reshape(1, -1)
    x_cnn = x_input.reshape(1, FEATURE_COUNT, 1)

    probabilities = []
    weights = []

    if "SVM" in models and hasattr(models["SVM"], "predict_proba"):
        probabilities.append(models["SVM"].predict_proba(x_input)[0])
        weights.append(0.178)

    if "Random Forest" in models and hasattr(models["Random Forest"], "predict_proba"):
        probabilities.append(models["Random Forest"].predict_proba(x_input)[0])
        weights.append(0.172)

    if "KNN" in models and hasattr(models["KNN"], "predict_proba"):
        probabilities.append(models["KNN"].predict_proba(x_input)[0])
        weights.append(0.174)

    if "Decision Tree" in models and hasattr(models["Decision Tree"], "predict_proba"):
        probabilities.append(models["Decision Tree"].predict_proba(x_input)[0])
        weights.append(0.169)

    if "BPNN" in models:
        probabilities.append(models["BPNN"].predict(x_input, verbose=0)[0])
        weights.append(0.151)

    if "CNN" in models:
        probabilities.append(models["CNN"].predict(x_cnn, verbose=0)[0])
        weights.append(0.155)

    if len(probabilities) == 0:
        raise ValueError("No available probability outputs for ensemble prediction.")

    probabilities = np.array(probabilities)
    weights = np.array(weights)

    weighted_proba = np.average(probabilities, axis=0, weights=weights)
    final_label = int(np.argmax(weighted_proba))

    return final_label, weighted_proba


def plot_spectral_curve(spectral_data, title):
    spectral_data = np.array(spectral_data, dtype=float)

    x_smooth = np.linspace(WAVELENGTHS.min(), WAVELENGTHS.max(), 400)
    y_smooth = np.interp(x_smooth, WAVELENGTHS, spectral_data)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)

    ax.set_facecolor("#eaeaea")

    for w in WAVELENGTHS:
        ax.axvline(
            w,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=0
        )

    ax.plot(
        x_smooth,
        y_smooth,
        color="seagreen",
        linewidth=2,
        zorder=2
    )

    ax.plot(
        WAVELENGTHS,
        spectral_data,
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor="seagreen",
        zorder=3
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral Value")
    ax.set_xticks(WAVELENGTHS)
    ax.set_xlim(400, 700)

    ax_bar = fig.add_axes([0.125, 0.08, 0.775, 0.06])
    gradient = np.linspace(0, 1, 600).reshape(1, -1)

    ax_bar.imshow(
        gradient,
        aspect="auto",
        cmap="nipy_spectral",
        extent=[WAVELENGTHS.min(), WAVELENGTHS.max(), 0, 1]
    )

    ax_bar.set_yticks([])
    ax_bar.set_xticks(WAVELENGTHS)

    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(bottom=0.28)

    return fig


def fig_to_pixmap(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    return pixmap


class SpectralDesktopApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Classification System")
        self.resize(1250, 850)

        self.models = load_models()
        self.current_fig = None
        self.batch_result_df = None

        if len(self.models) == 0:
            QMessageBox.critical(
                self,
                "Error",
                "No models found. Please make sure the ./models folder exists."
            )

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("Spectral Classification System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 30px; font-weight: bold; margin: 10px;"
        )
        main_layout.addWidget(title)

        subtitle = QLabel(
            "Input 8 spectral feature values or upload an Excel file to predict the class label."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 15px; color: #555; margin-bottom: 10px;")
        main_layout.addWidget(subtitle)

        content_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        model_layout = QVBoxLayout()

        self.model_box = QComboBox()
        self.model_box.addItems(list(self.models.keys()) + ["Ensemble"])
        self.model_box.setStyleSheet("font-size: 15px; padding: 6px;")
        model_layout.addWidget(self.model_box)

        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)

        # Manual prediction
        manual_group = QGroupBox("Manual Prediction")
        manual_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        manual_layout = QVBoxLayout()

        grid = QGridLayout()
        self.inputs = []

        for i, wavelength in enumerate(WAVELENGTHS):
            label = QLabel(f"{wavelength} nm")
            label.setStyleSheet("font-size: 13px;")

            input_box = QLineEdit()
            input_box.setText("0.000000")
            input_box.setStyleSheet("font-size: 14px; padding: 6px;")

            self.inputs.append(input_box)

            row = i // 2
            col = i % 2

            grid.addWidget(label, row, col * 2)
            grid.addWidget(input_box, row, col * 2 + 1)

        manual_layout.addLayout(grid)

        self.predict_button = QPushButton("Predict Single Sample")
        self.predict_button.setStyleSheet(
            "font-size: 16px; padding: 10px; background-color: #2E8B57; color: white;"
        )
        self.predict_button.clicked.connect(self.predict_single)
        manual_layout.addWidget(self.predict_button)

        self.result_label = QLabel("Predicted Label: -")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; margin: 10px;"
        )
        manual_layout.addWidget(self.result_label)

        self.proba_label = QLabel("Class Probabilities: -")
        self.proba_label.setWordWrap(True)
        self.proba_label.setAlignment(Qt.AlignCenter)
        self.proba_label.setStyleSheet("font-size: 14px; color: #333;")
        manual_layout.addWidget(self.proba_label)

        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.setStyleSheet("font-size: 15px; padding: 8px;")
        self.save_image_button.clicked.connect(self.save_image)
        manual_layout.addWidget(self.save_image_button)

        manual_group.setLayout(manual_layout)
        left_panel.addWidget(manual_group)

        # Batch prediction
        batch_group = QGroupBox("Batch Prediction")
        batch_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        batch_layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Excel File")
        self.upload_button.setStyleSheet("font-size: 15px; padding: 8px;")
        self.upload_button.clicked.connect(self.upload_excel)
        batch_layout.addWidget(self.upload_button)

        self.save_result_button = QPushButton("Save Prediction Results")
        self.save_result_button.setStyleSheet("font-size: 15px; padding: 8px;")
        self.save_result_button.clicked.connect(self.save_batch_results)
        batch_layout.addWidget(self.save_result_button)

        batch_group.setLayout(batch_layout)
        left_panel.addWidget(batch_group)

        left_panel.addStretch()

        # Plot
        plot_group = QGroupBox("Spectral Curve")
        plot_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        plot_layout = QVBoxLayout()

        self.image_label = QLabel("The spectral curve will be displayed here after prediction.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "font-size: 15px; color: #666; border: 1px solid #ddd; background-color: #fafafa;"
        )
        plot_layout.addWidget(self.image_label)

        plot_group.setLayout(plot_layout)
        right_panel.addWidget(plot_group)

        # Batch table
        table_group = QGroupBox("Prediction Results")
        table_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        table_layout = QVBoxLayout()

        self.table = QTableWidget()
        table_layout.addWidget(self.table)

        table_group.setLayout(table_layout)
        right_panel.addWidget(table_group)

        content_layout.addLayout(left_panel, 3)
        content_layout.addLayout(right_panel, 7)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def get_selected_model_name(self):
        return self.model_box.currentText()

    def predict_from_features(self, features):
        model_name = self.get_selected_model_name()
        x_input = np.array(features, dtype=float).reshape(1, -1)

        if model_name == "Ensemble":
            pred, proba = ensemble_predict(self.models, features)
            return pred, proba

        model = self.models[model_name]

        if model_name == "CNN":
            x_model = x_input.reshape(1, FEATURE_COUNT, 1)
            proba = model.predict(x_model, verbose=0)
            pred = int(np.argmax(proba, axis=1)[0])
            return pred, proba[0]

        if model_name == "BPNN":
            proba = model.predict(x_input, verbose=0)
            pred = int(np.argmax(proba, axis=1)[0])
            return pred, proba[0]

        pred = model.predict(x_input)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_input)[0]
        else:
            proba = None

        return pred, proba

    def predict_single(self):
        try:
            features = [float(box.text()) for box in self.inputs]

            if len(features) != FEATURE_COUNT:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Please enter exactly 8 spectral values."
                )
                return

            pred, proba = self.predict_from_features(features)

            pred_label = LABEL_MAP.get(int(pred), str(pred))
            self.result_label.setText(f"Predicted Label: {pred} ({pred_label})")

            if proba is not None:
                self.proba_label.setText(
                    f"Class Probabilities: {np.round(proba, 4)}"
                )
            else:
                self.proba_label.setText("Class Probabilities: Not available")

            self.current_fig = plot_spectral_curve(
                spectral_data=features,
                title=f"Spectral Curve - Predicted Label: {pred}"
            )

            pixmap = fig_to_pixmap(self.current_fig)
            self.image_label.setPixmap(
                pixmap.scaled(
                    820,
                    430,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            # === 在右下角显示结果 ===
            self.table.clear()
            self.table.setRowCount(1)
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Prediction", "Description"])
            pred_int = int(pred)
            pred_label = LABEL_MAP.get(int(pred), str(pred))

            self.table.setItem(0, 0, QTableWidgetItem(str(pred)))
            self.table.setItem(0, 1, QTableWidgetItem(pred_label))
            # Add colour warning
            if pred_int == 0:
                color = Qt.green
            elif pred_int == 1:
               color = Qt.yellow
            else:
                color = Qt.red

            for col in range(2):
                self.table.item(0, col).setBackground(color)
        except ValueError:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please enter valid numeric values for all 8 spectral features."
            )

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def upload_excel(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Excel File",
                "",
                "Excel Files (*.xlsx *.xls)"
            )

            if not file_path:
                return

            df = pd.read_excel(file_path, header=0)

            # The uploaded file contains only 8 spectral feature columns
            x_batch = df.iloc[:, :FEATURE_COUNT].values

            if x_batch.shape[1] != FEATURE_COUNT:
                QMessageBox.warning(
                    self,
                    "File Error",
                    f"The uploaded file must contain exactly 8 feature columns.\n"
                    f"Current feature number: {x_batch.shape[1]}"
                )
                return

            preds = []

            for row in x_batch:
                pred, _ = self.predict_from_features(row)
                preds.append(pred)

            result_df = pd.DataFrame({
                "Sample": range(1, len(preds) + 1),
                "Prediction": preds,
                "Description": [
                    LABEL_MAP.get(int(p), str(p)) for p in preds
                ]
            })

            self.batch_result_df = result_df
            self.show_dataframe(result_df)

            QMessageBox.information(
                self,
                "Success",
                "Batch prediction completed successfully."
            )

        except Exception as e:
            QMessageBox.critical(self, "Batch Prediction Error", str(e))

    def show_dataframe(self, df):
        self.table.clear()
        self.table.setRowCount(min(len(df), 100))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i in range(min(len(df), 100)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.table.setItem(i, j, item)

        # Add colour warning only when Prediction column exists
        if "Prediction" in df.columns:
            for i in range(min(len(df), 100)):
                pred = int(df.iloc[i]["Prediction"])

                if pred == 0:
                    color = Qt.green
                elif pred == 1:
                    color = Qt.yellow
                else:
                    color = Qt.red

                for j in range(len(df.columns)):
                    self.table.item(i, j).setBackground(color)

    def save_batch_results(self):
        if self.batch_result_df is None:
            QMessageBox.warning(
                self,
                "No Results",
                "Please run batch prediction first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Prediction Results",
            "prediction_results.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".xlsx"):
                self.batch_result_df.to_excel(file_path, index=False)
            else:
                self.batch_result_df.to_csv(file_path, index=False, encoding="utf-8-sig")

            QMessageBox.information(
                self,
                "Saved",
                f"Prediction results saved to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def save_image(self):
        if self.current_fig is None:
            QMessageBox.warning(
                self,
                "No Image",
                "Please run single-sample prediction first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectral Curve Image",
            "spectral_curve.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )

        if not file_path:
            return

        try:
            self.current_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            QMessageBox.information(
                self,
                "Saved",
                f"Image saved to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectralDesktopApp()
    window.show()
    sys.exit(app.exec())
     