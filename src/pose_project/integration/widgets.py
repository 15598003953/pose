"""Image canvases with calibrated, letterbox-aware rectangle editing."""

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget


class VideoCanvas(QWidget):
    rectangle_drawn = pyqtSignal(str, list)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.image = QImage()
        self.overlays = {}
        self.target = None
        self.origin = self.endpoint = None
        self.setMinimumSize(240, 210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, bgr):
        height, width = bgr.shape[:2]
        rgb = bgr[:, :, ::-1].copy()
        self.image = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format_RGB888).copy()
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def begin_draw(self, target):
        self.target = target
        self.setCursor(Qt.CrossCursor)

    def _point(self, position):
        rect = self.image_rect()
        if rect.isEmpty():
            return None
        return QPointF(
            max(0, min(self.image.width(), (position.x() - rect.x()) / rect.width()
                       * self.image.width())),
            max(0, min(self.image.height(), (position.y() - rect.y()) / rect.height()
                       * self.image.height())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.target and not self.image.isNull():
            if self.image_rect().contains(event.pos()):
                self.origin = self.endpoint = self._point(event.pos())
                self.update()

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            self.endpoint = self._point(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.origin is not None:
            end = self._point(event.pos())
            box = [int(min(self.origin.x(), end.x())), int(min(self.origin.y(), end.y())),
                   int(max(self.origin.x(), end.x())), int(max(self.origin.y(), end.y()))]
            if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
                self.rectangle_drawn.emit(self.target, box)
            self.origin = self.endpoint = self.target = None
            self.unsetCursor()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171c20"))
        rect = self.image_rect()
        if self.image.isNull():
            painter.setPen(QColor("#a8b3ba"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.title)
            return
        painter.drawImage(rect, self.image)
        painter.save()
        painter.translate(rect.topLeft())
        painter.scale(rect.width() / self.image.width(), rect.height() / self.image.height())
        for label, (box, color) in self.overlays.items():
            painter.setPen(QPen(QColor(color), 2))
            painter.drawRect(QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1]))
            painter.drawText(QPointF(box[0] + 5, box[1] + 18), label)
        if self.origin is not None:
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.DashLine))
            painter.drawRect(QRectF(self.origin, self.endpoint).normalized())
        painter.restore()
