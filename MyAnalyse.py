import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import os
import shutil
import evaluate
import numpy as np
from transformers import AutoTokenizer
import nltk

def analyze_predictions(testset_path, test_answer_path):
    """
    Updated version of the analyze_predictions function to include accuracy and make certain annotations more prominent.

    Parameters:
    - testset_path (str): Path to the JSON file containing the actual answers.
    - test_answer_path (str): Path to the JSON file containing the model's predictions.

    Returns:
    - A bar chart visualizing the analysis of predictions vs actual answers with metrics.
    """

    # Load the files
    with open(testset_path, "r") as f:
        testset = json.load(f)

    with open(test_answer_path, "r") as f:
        test_answer = json.load(f)

    # Convert the model's answers to numerical format for comparison
    model_answers = {key.replace("question_", ""): 0 if "The answer is (A)." in value else 1 for key, value in
                     test_answer.items()}
    actual_answers = {key: value["answer"] for key, value in testset.items()}

    # Calculate TP, TN, FP, FN
    TP = sum([model_answers[key] == 0 and actual_answers[key] == 0 for key in model_answers])
    FP = sum([model_answers[key] == 0 and actual_answers[key] == 1 for key in model_answers])
    TN = sum([model_answers[key] == 1 and actual_answers[key] == 1 for key in model_answers])
    FN = sum([model_answers[key] == 1 and actual_answers[key] == 0 for key in model_answers])

    # Calculate precision, recall, F1 score, and accuracy
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Data for plotting
    labels = ['Predicted NO, Actual YES', 'Predicted YES, Actual NO', 'Total YES answers', 'Total NO answers']
    values = [FN, FP, TP + FN, FP + TN]

    # Plotting with annotations
    plt.figure(figsize=(16, 7))
    bars = plt.barh(labels, values, color=['red', 'blue', 'green', 'purple'])

    # Annotate the bars with the actual values
    for bar in bars:
        width = bar.get_width()
        plt.text(width - 0.05 * max(values), bar.get_y() + bar.get_height() / 2,
                 f'{width}', ha='center', va='bottom', color='white', fontsize=12)

    # Annotate with precision, recall, F1 score, and accuracy using larger font sizes
    metrics_text = [
        f'Precision: {precision:.4f}',
        f'Recall: {recall:.4f}',
        f'F1 Score: {f1_score:.4f}',
        f'Accuracy: {accuracy:.4f}'
    ]
    for idx, text in enumerate(metrics_text):
        plt.text(1.15 * max(values), bars[-1].get_y() - idx * bars[1].get_height() / 1.5, text, va='bottom', fontsize=14)

    plt.xlabel('Number of Questions')
    plt.title('Analysis of Predictions vs Actual Answers with Metrics')
    plt.gca().invert_yaxis()  # Display the bars in the order of the labels
    plt.xlim(0, 1.4 * max(values))
    plt.show()

def compute_metrics_rougel(_generate_path, _target_path, _pretrained_model_path):
    def postprocess_text(_preds, _labels):
        _preds = [pred.strip() for pred in _preds]
        _labels = [label.strip() for label in _labels]
        _preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in _preds]
        _labels = ["\n".join(nltk.sent_tokenize(label)) for label in _labels]
        return _preds, _labels

    metric = evaluate.load("./rouge.py")
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model_path)
    with open(_generate_path, "r", encoding="utf-8") as GFile:
        generated_data = json.load(GFile)
    with open(_target_path, "r", encoding="utf-8") as TFile:
        target_data = json.load(TFile)

    generated_data = [solution for question, solution in generated_data.items()]
    # print(generated_data)
    target_data = [context['solution'] for question, context in target_data.items()]
    # print(target_data)
    if len(generated_data) != len(target_data):
        raise ValueError("The length of generated_data and target_data does not match.")
    preds, targets = postprocess_text(generated_data, target_data)

    preds_token = tokenizer.batch_encode_plus(
        preds,
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets_token = tokenizer.batch_encode_plus(
        targets,
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    result = metric.compute(predictions=preds, references=targets, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}


    preds_len = [np.count_nonzero(pred_token != tokenizer.pad_token_id) for pred_token in preds_token['input_ids']]
    targets_len = [np.count_nonzero(target_token != tokenizer.pad_token_id) for target_token in targets_token['input_ids']]
    print(np.mean(preds_len))
    print(np.mean(targets_len))
    # result["gen_toekn_len"] = np.mean(prediction_lens)
    print(result)

def identify_misclassifications(testset_path, test_answer_path):
    """
    Identify the questions where the model made incorrect predictions.

    Parameters:
    - testset_path (str): Path to the JSON file containing the actual answers.
    - test_answer_path (str): Path to the JSON file containing the model's predictions.

    Returns:
    - A dictionary containing two lists:
      1. "yes_predicted_as_no": Questions where the actual answer was YES but the model predicted NO.
      2. "no_predicted_as_yes": Questions where the actual answer was NO but the model predicted YES.
    """

    # Load the files
    with open(testset_path, "r") as f:
        testset = json.load(f)

    with open(test_answer_path, "r") as f:
        test_answer = json.load(f)

    # Convert the model's answers to numerical format for comparison
    model_answers = {key.replace("question_", ""): 0 if "The answer is (A)." in value else 1 for key, value in
                     test_answer.items()}
    actual_answers = {key: value["answer"] for key, value in testset.items()}

    # Identify the misclassifications
    yes_predicted_as_no = [key for key in model_answers if model_answers[key] == 0 and actual_answers[key] == 1]
    no_predicted_as_yes = [key for key in model_answers if model_answers[key] == 1 and actual_answers[key] == 0]

    print(f"Predicted Yes, Actual No ({len(yes_predicted_as_no)}):\n{yes_predicted_as_no}\n")
    print(f"Predicted No, Actual Yes ({len(no_predicted_as_yes)}):\n{no_predicted_as_yes}")
    return {"Predicted_Yes_Actual_No": yes_predicted_as_no, "Predicted_No_Actual_Yes": no_predicted_as_yes}

def identify_misclassified_images_detailed(_testset_path, _test_answer_path, _outfile_path):
    """
    Redesigned function to identify the images for which the model made incorrect predictions on their corresponding questions.
    This function provides a detailed output with question-level misclassifications per image.

    Parameters:
    - testset_path (str): Path to the JSON file containing the actual answers and image IDs.
    - test_answer_path (str): Path to the JSON file containing the model's predictions.

    Returns:
    - A list of dictionaries, each containing image ID and its corresponding misclassified questions.
    """

    # Load the files
    with open(_testset_path, "r") as f:
        testset = json.load(f)

    with open(_test_answer_path, "r") as f:
        test_answer = json.load(f)

    # Convert the model's answers to numerical format for comparison
    model_answers = {key.replace("question_", ""): 0 if "The answer is (A)." in value else 1 for key, value in
                     test_answer.items()}

    # Identify the misclassifications and categorize them by image
    misclassification_by_image = {}

    for key in model_answers:
        image_id = testset[key]["image"]
        actual_answer = testset[key]["answer"]
        predicted_answer = model_answers[key]

        # Check for misclassification
        if actual_answer != predicted_answer:
            if image_id not in misclassification_by_image:
                misclassification_by_image[image_id] = {}

            # Categorize the misclassification
            misclassification_type = "Predicted_Yes_Actual_No" if actual_answer == 1 else "Predicted_No_Actual_Yes"
            misclassification_by_image[image_id][key] = misclassification_type

    # Format the output as per the requested structure
    output = []
    for image, questions in misclassification_by_image.items():
        image_entry = {"img_id": image}
        image_entry.update(questions)
        output.append(image_entry)
    print(output)
    with open(_outfile_path, "w", encoding="utf-8") as OutFile:
        json.dump(output, OutFile, indent=4, ensure_ascii=False)
    # return output

def calculate_and_plot_venn(list1, list2, list3, _output_file_path):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    set3 = set(list3)

    # Calculate intersections and unique elements
    common_elements = set1 & set2 & set3
    unique_in_set1 = set1 - common_elements
    unique_in_set2 = set2 - common_elements
    unique_in_set3 = set3 - common_elements
    in_set1_and_set2 = (set1 & set2) - set3
    in_set1_and_set3 = (set1 & set3) - set2
    in_set2_and_set3 = (set2 & set3) - set1

    # Create the Venn diagram
    plt.figure(figsize=(8, 8))
    venn_diagram = venn3([set1, set2, set3], ('Unifiedqa_T5_Base', 'Flan_T5_Large', 'Large_T5'))
    plt.show()

    # Prepare data for output
    intersections = {
        'Unifiedqa_T5_Base ∩ Flan_T5_Large ∩ Large_T5': list(common_elements),
        'Unifiedqa_T5_Base ∩ Flan_T5_Large - Large_T5': list(in_set1_and_set2),
        'Unifiedqa_T5_Base ∩ Large_T5 - Flan_T5_Large': list(in_set1_and_set3),
        'Flan_T5_Large ∩ Large_T5 - Unifiedqa_T5_Base': list(in_set2_and_set3),
        'Unifiedqa_T5_Base - (Flan_T5_Large ∪ Large_T5)': list(unique_in_set1),
        'Flan_T5_Large - (Unifiedqa_T5_Base∪ Large_T5)': list(unique_in_set2),
        'Large_T5 - (Unifiedqa_T5_Base ∪ Flan_T5_Large)': list(unique_in_set3)
    }

    # Sort each list of elements for better readability
    for key in intersections:
        intersections[key].sort(key=int)  # Sorting as integers to maintain natural order

    with open(_output_file_path, "w", encoding="utf-8") as OutFile:
        json.dump(intersections, OutFile, indent=4, ensure_ascii=False)
    # return intersections

def plot_venn_and_save_annotations(sets_dict, output_file):
    """
    Plots a Venn diagram based on the input dictionary and saves annotations as a JSON file.
    The dictionary can have up to 3 sets.
    """
    # Convert lists to sets
    sets = {k: set(v) for k, v in sets_dict.items()}
    labels = list(sets.keys())
    data = list(sets.values())

    # Determine the number of sets and plot accordingly
    plt.figure(figsize=(8, 8))
    if len(sets) == 2:
        venn2(data, labels)
    elif len(sets) == 3:
        venn3(data, labels)
    else:
        raise ValueError("This function only supports 2 or 3 sets for Venn diagrams.")

    title = ' vs. '.join(labels)
    plt.title(title)
    plt.show()

    # Prepare annotations for unique and shared elements
    annotations = {}
    for label in labels:
        unique_elements = sets[label] - set().union(*[sets[l] for l in labels if l != label])
        annotations[f"Unique to {label}"] = list(unique_elements)

    shared_elements = set.intersection(*[sets[label] for label in labels])
    annotations["Shared Elements"] = list(shared_elements)

    # Save annotations to a JSON file
    with open(output_file, 'w') as file:
        json.dump(annotations, file, indent=4)

    return output_file

def copy_misclassified_images(_json_path, _source_dir, _destination_dir, simulate=False):
    """
    Copies images listed in a JSON file from a source directory to a destination directory.

    :param json_path: Path to the JSON file containing image IDs.
    :param source_dir: Path to the source directory where images are located.
    :param destination_dir: Path to the destination directory where images will be copied.
    :param simulate: If True, simulate the existence of image files. If False, assume files exist.
    """
    # Load the JSON file to get the list of image IDs
    with open(_json_path, 'r') as file:
        data = json.load(file)

    # Extracting the image IDs
    img_ids = [item['img_id'] for item in data]

    # Create the destination directory if it doesn't exist
    os.makedirs(_destination_dir, exist_ok=True)

    # If simulate is True, create dummy image files in the source directory
    if simulate:
        os.makedirs(_source_dir, exist_ok=True)
        for img_id in img_ids:
            with open(os.path.join(_source_dir, img_id), 'w') as f:
                f.write("")  # Creating an empty file to simulate the image

    # Copy image files to the destination directory
    for img_id in img_ids:
        source_file = os.path.join(_source_dir, img_id)
        destination_file = os.path.join(_destination_dir, img_id)

        # Check if the source file exists before copying
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        else:
            if not simulate:
                print(f"Warning: The file {source_file} does not exist and will not be copied.")

    # Return the list of copied files for verification
    return os.listdir(_destination_dir)

def plot_category_distribution_from_json(_json_file_path, image_organ_key='image_organ', categories=None, explode=None, title='Category Distribution'):
    """
    Loads a JSON file, counts the occurrences of specified categories within the 'image_organ' key, and plots a pie chart.

    :param _json_file_path: The file path of the JSON file containing the dataset.
    :param image_organ_key: The key in the dataset where the category is specified.
    :param categories: A list of categories to count. If None, counts all categories found.
    :param explode: A tuple indicating which slices to "explode" or stand out in the pie chart.
    :param title: The title for the pie chart.
    """
    # Load the json data
    with open(_json_file_path) as json_file:
        data = json.load(json_file)

    # Initialize the category counts
    if categories is None:
        category_counts = {}
    else:
        category_counts = {category: 0 for category in categories}

    # Count occurrences of each category
    for entry in data:
        if entry['answer_type'].upper() == 'CLOSED':
            category = entry.get(image_organ_key, '').upper()
            if categories is None or category in category_counts:
                category_counts[category] = category_counts.get(category, 0) + 1

    # Labels and sizes for the pie chart
    labels = category_counts.keys()
    sizes = category_counts.values()
    colors = ['gold', 'lightskyblue', 'lightcoral']

    # If no explode tuple is provided, do not explode any slices
    if explode is None:
        explode = (0,) * len(labels)

    # Function to format the label with both percentage and absolute number
    def absolute_value(val):
        a = int(round(val / 100. * sum(sizes)))
        return f"{a}\n({val:.1f}%)"

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=absolute_value, shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.show()

    return category_counts

def analyze_and_plot_organ_answers(json_path):
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Define the organ and answer categories
    organs = ['CHEST', 'ABD', 'HEAD']
    answers = ['yes', 'no']

    # Initialize a dictionary to hold the count for each combination
    organ_answer_combinations = {organ: {answer: 0 for answer in answers} for organ in organs}

    # Count the occurrences of each combination
    for item in data.values():
        organ = item['image_organ']
        answer = answers[item['answer']]  # 'yes' for 0, 'no' for 1
        if organ in organ_answer_combinations:
            organ_answer_combinations[organ][answer] += 1

    # Prepare data for plotting
    counts = {organ: [organ_answer_combinations[organ][answer] for answer in answers] for organ in organs}

    # Create the bar plot
    fig, ax = plt.subplots()
    index = range(len(organs))
    bar_width = 0.35

    bars1 = ax.bar(index, [counts[organ][0] for organ in organs], bar_width, color='skyblue', label='Yes')
    bars2 = ax.bar([i + bar_width for i in index], [counts[organ][1] for organ in organs], bar_width,
                   color='lightgreen', label='No')

    # Add counts above bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Organ')
    ax.set_ylabel('Count')
    ax.set_title('Counts by organ and answer')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(organs)
    ax.legend()

    plt.show()

    return organ_answer_combinations


if __name__ == "__main__":
    # test_file_path = "/path/to/mm_cot/VQA-RAD_ByUs/without_open/testset.json"
    # sc_answer_file_path = "/path/to/mm_cot/experiments/unifiedqa_caption_sc.json"
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=sc_answer_file_path
    # )

    # test_file_path = "/path/to/mm_cot/VQA-RAD_ByUs/without_open/testset.json"
    # genimi_answer_file_path = "/path/to/mm_cot/Genimi/rad_answer.json"
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=genimi_answer_file_path
    # )

    # test_file_path = "/path/to/mm_cot/VQA-SLAKE_ByUs/without_open/test.json"
    # genimi_answer_file_path = "/path/to/mm_cot/Genimi/slake_answer.json"
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=genimi_answer_file_path
    # )
    dataset = "rad"
    test_list = {'rad': '/path/to/mm_cot/VQA-RAD_ByUs/without_open/testset.json',
                 'slake': '/path/to/mm_cot/VQA-SLAKE_ByUs/without_open/test.json'}
    analyze_predictions(
        testset_path=test_list[dataset],
        test_answer_path=f'/path/to/mm_cot/paper_result/{dataset}_QM_RA/test_answer.json'
    )


    # test_file_path = '/path/to/mm_cot/VQA-RAD_ByUs/without_open/cap_testset.json'
    # caption_answer_file_path = '/path/to/mm_cot/experiments/detr_15/Answer/test_answer_4560.json'
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=caption_answer_file_path
    # )

    # test_file_path = '/path/to/mm_cot/VQA-SLAKE_ByUs/without_open/test.json'
    # caption_answer_file_path = '/path/to/mm_cot/slake_experiments/detr_0/Answer/test_answer.json'
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=caption_answer_file_path
    # )

    # test_file_path = '/path/to/mm_cot/VQA-SLAKE_ByUs/without_open/test.json'
    # answer_file_path = '/path/to/mm_cot/slake_experiments/detr_3/Answer/test_answer_1701.json'
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=answer_file_path
    # )

    # unifiedqa_t5_test_answer_file_path = "/path/to/mm_cot/experiments/detr_100_5e-05_4_zero/Answer/test_answer.json"
    # flan_large_t5_test_answer_file_four_path = "/path/to/mm_cot/experiments/detr_four/Answer/test_answer.json"
    # large_t5_test_answer_file_path = '/path/to/mm_cot/experiments/detr_five/Answer/test_answer.json'
    # flan_large_t5_test_answer_file_six_path = "/path/to/mm_cot/experiments/detr_six/Answer/test_answer.json"
    # analyze_predictions(test_file_path, flan_large_t5_test_answer_file_path)
    # flan_large_t5_seg_test_answer_file_path = "/path/to/mm_cot/experiments/detr_eight/Answer/test_answer.json"
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=unifiedqa_t5_test_answer_file_path
    # )

    # print("Dataset: RAD, Model: Unifiedqa-T5-Base, w/o Caption")
    # test_file_path = "/path/to/mm_cot/VQA-RAD_ByUs/without_open/testset.json"
    # unifiedqa_t5_path = "/path/to/mm_cot/experiments/detr_100_5e-05_4_zero/Answer/test_answer.json"
    # cap_unifiedqa_t5_path = "/path/to/mm_cot/experiments/detr_14/Answer/test_answer.json"
    # unifiedqa_t5_error_ids = identify_misclassifications(test_file_path, unifiedqa_t5_path)
    # cap_unifiedqa_t5_error_ids = identify_misclassifications(test_file_path, cap_unifiedqa_t5_path)
    # false_positive = {
    #     'No Caption': unifiedqa_t5_error_ids['Predicted_Yes_Actual_No'],
    #     'With Caption': cap_unifiedqa_t5_error_ids['Predicted_Yes_Actual_No']
    # }
    # false_negative = {
    #     'No Caption': unifiedqa_t5_error_ids['Predicted_No_Actual_Yes'],
    #     'With Caption': cap_unifiedqa_t5_error_ids['Predicted_No_Actual_Yes']
    # }
    # plot_venn_and_save_annotations(
    #     sets_dict=false_positive,
    #     output_file='/path/to/mm_cot/experiments/unifiedqa_t5_base_cap_fp.json'
    # )
    # plot_venn_and_save_annotations(
    #     sets_dict=false_negative,
    #     output_file='/path/to/mm_cot/experiments/unifiedqa_t5_base_cap_fn.json'
    # )


    # print("unifiedqa_t5:")
    # unifiedqa_t5_error_ids = identify_misclassifications(test_file_path, unifiedqa_t5_test_answer_file_path)
    # print("\nflan_large_t5:")
    # flan_large_t5_error_ids = identify_misclassifications(test_file_path, flan_large_t5_test_answer_file_four_path)
    # print("\nflan_large_t5_seg:")
    # flan_large_t5_seg_error_ids = identify_misclassifications(test_file_path, flan_large_t5_seg_test_answer_file_path)
    # print("\nlarge_t5:")
    # large_t5_error_ids = identify_misclassifications(test_file_path, large_t5_test_answer_file_path)
    #
    # venn_type = ["Predicted_Yes_Actual_No", "Predicted_No_Actual_Yes"]
    # a = 1
    # calculate_and_plot_venn(unifiedqa_t5_error_ids[venn_type[a]],
    #                         flan_large_t5_error_ids[venn_type[a]],
    #                         large_t5_error_ids[venn_type[a]],
    #                         f'Venn_result_{venn_type[a]}.json')
    # print("\nunifiedqa_t5:")
    # identify_misclassified_images_detailed(test_file_path, unifiedqa_t5_test_answer_file_path,
    #                                        'unifiedqa_t5_misclassified_images.json')
    # print("\nflan_large_t5:")
    # identify_misclassified_images_detailed(test_file_path, flan_large_t5_test_answer_file_path,
    #                                        'flan_large_t5_misclassified_images.json')
    # print("\nlarge_t5:")
    # identify_misclassified_images_detailed(test_file_path, large_t5_test_answer_file_path,
    #                                        'large_t5_misclassified_images.json')

    # copy_misclassified_images(
    #     _json_path='/path/to/mm_cot/unifiedqa_t5_misclassified_images.json',
    #     _source_dir='/path/to/mm_cot/data_rad/images',
    #     _destination_dir='/path/to/mm_cot/Error/unifiedqa_t5_base/'
    # )
    # copy_misclassified_images(
    #     _json_path='/path/to/mm_cot/flan_large_t5_misclassified_images.json',
    #     _source_dir='/path/to/mm_cot/data_rad/images',
    #     _destination_dir='/path/to/mm_cot/Error/flan_large_t5/'
    # )
    # copy_misclassified_images(
    #     _json_path='/path/to/mm_cot/large_t5_misclassified_images.json',
    #     _source_dir='/path/to/mm_cot/data_rad/images',
    #     _destination_dir='/path/to/mm_cot/Error/large_t5/'
    # )

    # data_rad_json_path = '/path/to/mm_cot/data_rad/trainset.json'
    # plot_category_distribution_from_json(
    #     _json_file_path=data_rad_json_path,
    #     categories=["ABD", "CHEST", "HEAD"],
    #     explode=(0.1, 0, 0),
    #     title="Distribution of Questions by Category"
    # )

    # train_json_path = '/path/to/mm_cot/VQA-RAD_ByUs/without_open/trainset.json'
    # test_json_path = '/path/to/mm_cot/VQA-RAD_ByUs/without_open/testset.json'
    # result = analyze_and_plot_organ_answers(test_json_path)
    # print(result)

    # false_positive = {
    #     'flan_large_t5': flan_large_t5_error_ids['Predicted_Yes_Actual_No'],
    #     'flan_large_t5_seg': flan_large_t5_seg_error_ids['Predicted_Yes_Actual_No']
    # }
    # false_negative = {
    #     'flan_large_t5': flan_large_t5_error_ids['Predicted_No_Actual_Yes'],
    #     'flan_large_t5_seg': flan_large_t5_seg_error_ids['Predicted_No_Actual_Yes']
    # }
    # plot_venn_and_save_annotations(false_positive, '/path/to/mm_cot/venn_annotations_fp.json')
    # plot_venn_and_save_annotations(false_negative, '/path/to/mm_cot/venn_annotations_fn.json')
    #


    # ============SLAKE======================
    # test_file_path = "/path/to/mm_cot/VQA-SLAKE_ByUs/without_open/test.json"
    # t5_large_answer = "/path/to/mm_cot/slake_experiments/detr_2/Answer/test_answer.json"
    # analyze_predictions(
    #     testset_path=test_file_path,
    #     test_answer_path=t5_large_answer
    # )
    

    # generate_file = f"/path/to/mm_cot/experiments/detr_15/Rational/test_solution.json"
    # target_file = f"/path/to/mm_cot/VQA-RAD_ByUs/without_open/cap_testset.json"
    # compute_metrics_rougel(
    #     _generate_path=generate_file,
    #     _target_path=target_file,
    #     # _pretrained_model_path="/path/to/mm_cot/slake_experiments/detr_0/Rational/checkpoint-24300"
    #     _pretrained_model_path="/path/to/mm_cot/src/unifiedqa-t5-base"
    # )








