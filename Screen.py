import PySimpleGUI as sg

class Screen:

    def __init__(self):
        self.setFilename(sg.popup_get_file('Importe a base de dados (.csv)', 'Importação da Base de Dados', file_types=(("Csv Files", "*.csv"),)))

    def getFilename(self):
        return self.filename
    
    def setFilename(self, _filename):
        self.filename = _filename
    