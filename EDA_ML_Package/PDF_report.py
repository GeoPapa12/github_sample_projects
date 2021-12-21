from datetime import datetime
import os
import io
import warnings

# PDF Report building
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

warnings.filterwarnings('ignore')


class PDF_reporting():
    # Story = []

    def __init__(self, Story=[], doc_title=""):
        self.Story = Story
        self.doc_title = doc_title
        self.now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self.styles = getSampleStyleSheet()
        # self.doc = SimpleDocTemplate("ML analysis " + self.nowTitle + "_" + self.doc_title + ".pdf", pagesize=letter,
        #                             rightMargin=inch/2, leftMargin=inch/2,
        #                             topMargin=25.4, bottomMargin=12.7)

        if str(self).find('ML_models') != -1:
            self.add_text("ML Analysis", style="Heading1", fontsize=24)
            self.add_text(self.now)

    def directory_one_level_up(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def add_text(self, text, style="Normal", fontsize=12, page_break=0):
        """ Adds text with some spacing around it to  PDF report

        Parameters
        ----------
        text : (str) The string to print to PDF
        style : (str) The reportlab style
        fontsize : (int) The fontsize for the text
        """
        if page_break == 1:
            self.Story.append(PageBreak())

        self.Story.append(Spacer(1, 4))
        ptext = "<font size={}>{}</font>".format(fontsize, text)
        self.Story.append(Paragraph(ptext,  self.styles[style]))
        self.Story.append(Spacer(1, 4))

    def table_in_PDF(self, df_results):
        """ Adds style to table to be printed in pdf

        Parameters
        ----------
        table : (list) table to be printed in pdf

        Output: table (list)
        """

        colNames = df_results.columns.to_list()
        table = df_results.values.tolist()
        table.insert(0, colNames)
        table = Table(table, hAlign='LEFT')

        table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('INNERGRID', (0, 0), (-1, -1), 0.50, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ]))

        self.add_text("")
        self.Story.append(table)
        self.add_text("")

        return table

    def image_in_PDF(self, plot, x=7, y=2.5):

        buf = io.BytesIO()
        plot.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        # you'll want to close the figure once its saved to buffer
        if 'Figure' in str(type(plot)) is False:
            plot.close()

        self.Story.append(Image(buf, x*inch, y*inch))
        return buf

    def generate_report(self, docTitle):
        """ Buids the PDF report

        Parameters
        ----------
        -

        Output: -
        """
        nowTitle = datetime.now().strftime("%d_%m_%Y %H-%M-%S")

        self.doc = SimpleDocTemplate(str(docTitle) + "_" + nowTitle + "_" + self.doc_title + ".pdf", pagesize=letter,
                                     rightMargin=inch/2, leftMargin=inch/2,
                                     topMargin=25.4, bottomMargin=12.7)

        self.doc.build(self.Story)
