import json

class InputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.options = ['A', 'B']
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.choice_text = self.get_choice_text()
        self.solution_text = self.get_solution_text()

    def get_choice_text(self):
        choices = self.problem['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(self.options[i], c))
        choice_txt = " ".join(choice_list)
        return choice_txt

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return "(" + self.options[self.problem['answer']] + ")"

    def get_solution_text(self):
        return self.problem['solution']

    def get_target(self, _rational=True):
        if _rational:
            return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        else:
            return f"The answer is {self.answer_text}."

    def get_input(self, _rational=True, _caption=True, _answer_first=False):
        prior_text = f"Question: {self.question_text}\nOptions: {self.choice_text}\n"
        if _caption:
            prior_text = prior_text + f"Caption: {self.problem['caption']}\n"

        if _rational and not _answer_first:
            return prior_text + "Solution:"
        else:
            return prior_text + "Answer:"

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid _dataset value: {_dataset}. The value must be 'rad' or 'slake'.")

path = '/home/chenyizhou/mm_cot/VQA-RAD_ByUs/without_open/V2/cap_testset.json'
with open(path, 'r', encoding='utf-8') as File:
    data = json.load(File)
for item in data:
    a = InputAndTargetAndImg(data[item])
    break
print(a.get_input())