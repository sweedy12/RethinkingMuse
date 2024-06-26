import json
import os

class PurposeReader:
    START_IND = len("The purpose of the patent is to provide") + 1

    def read_json_file(self,fname):
        with open(fname) as f:
            line = f.readlines()[0]
            d = json.loads(line)
            return d

    def extract_purpose_from_text(self,text):
        last_ind = text.find(".")
        purp = text[self.START_IND:last_ind]
        return purp



    def update_dict_from_dict(self,base, d):
        for x in d:
            id  = x["id"]
            # if id not in base:
            #     base[id] = []
            base[id] = self.extract_purpose_from_text(x["response"])
        return base


    def create_purpose_dict(self,dir):
        purpose_dict = {}
        for fname in os.listdir(dir):
            d = self.read_json_file(dir+fname)
            purpose_dict = self.update_dict_from_dict(purpose_dict, d)
        return purpose_dict




# # text = "The purpose of the patent is to provide a device for inserting an object into a cavity. The context of the patent is medical devices."
# # extract_purpose_from_text(text)
# # fname = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\nwq_patent_tags_0_1000_2021_12_05_06_27_37.json"
# # read_json_file(fname)
# dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
# PR = PurposeReader()
# purpose_dict = PR.create_purpose_dict(dir)
# x = 1
