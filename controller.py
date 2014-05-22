# Allow access to command-line arguments
import sys

# Import the core and GUI elements of Qt
from PySide.QtCore import *
from PySide.QtGui import *
# Create the QApplication object
qt_app = QApplication(sys.argv)


class AbsolutePositioningExample(QWidget):

    def __init__(self):

        QWidget.__init__(self)

        self.setWindowTitle('Dynamic Greeter')
        self.setMinimumWidth(400)

        self.layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        self.salutations = ['Ahoy',
                            'Good day',
                            'Hello',
                            'Heyo',
                            'Hi',
                            'Salutations',
                            'Wassup',
                            'Yo']

        # Create and fill the combo box to choose the salutation
        self.salutation = QComboBox(self)
        self.salutation.addItems(self.salutations)

        self.form_layout.addRow('Salutation:', self.salutation)

        self.recipient = QLineEdit(self)
        self.recipient.setPlaceholderText("e.g 'world or 'Matey'")

        self.form_layout.addRow('Recipient:', self.recipient)

        self.greeting = QLabel('', self)
        self.form_layout.addRow('Greeting:', self.greeting)

        self.layout.addLayout(self.form_layout)

        self.layout.addStretch(1)

        self.button_box = QHBoxLayout()

        self.build_button = QPushButton('Build Greeting', self)

        self.button_box.addWidget(self.build_button)

        self.layout.addLayout(self.button_box)

        self.setLayout(self.layout)

    def run(self):
        self.show()
        qt_app.exec_()

app = AbsolutePositioningExample()
app.run()
