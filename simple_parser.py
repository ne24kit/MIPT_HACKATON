from bs4 import BeautifulSoup
import markdown

class File_parser():
    __slots__ = ('md_files', 'path', 'parsed_res', 'markdown_parse')
    def __init__(self, path = './'):
        self.path = path
        self.md_files = []
        self.parsed_res = []
        self.markdown_parse = markdown.Markdown()

    def new_files_parse(self, files_name = []):
        self.md_files = files_name
        self.parsed_res = [0] * len(self.md_files)
        self.parse_func()

    def parse_func(self):
        for file_ind, file_name in enumerate(self.md_files):
            with open(self.path + file_name, 'r'  ,encoding='utf-8') as f:
                parse = BeautifulSoup(self.markdown_parse.convert(f.read()), features='lxml')

            text_list = parse.find_all('p')

            self.parsed_res[file_ind] = {}
            for ind, paragraph in enumerate(text_list):
                self.parsed_res[file_ind][ind+1] = paragraph.text
            self.markdown_parse.reset()

