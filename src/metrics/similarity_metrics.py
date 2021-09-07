from typing import List, Tuple

def split_name(prediction: str) -> Tuple[str, int]:
    if len(prediction.split("_")) > 1:
        return prediction.split("_")[0], int(prediction.split("_")[1])
    else:
        return prediction, 0
    

def is_correctly_retrieved(predictions: List[str], ground_truth: List[str]) -> bool:

    top_1_predicted, deg = split_name(predictions[0])

    if top_1_predicted in ground_truth:
        #print(top_1_predicted,":",deg)
        return True

    return False


def is_category_correctly_retrieved(predictions: str, ground_truth: str) -> bool:
    if predictions == ground_truth:
        return True
    return False


def is_correctly_ranked(prediction: str, rank, ground_truth: List[str]) -> bool:
    if prediction == ground_truth[rank]:
        return True

    return False

def count_correctly_ranked_predictions(predictions: List[str], ground_truth: List[str]) -> int:
    num_ranked_models = len(ground_truth)  # 3 or less
    top_n_predicted = predictions[:num_ranked_models]

    correctly_ranked = 0
    total = num_ranked_models
    for rank, prediction in enumerate(top_n_predicted): # TODO, fix: this metric is not reasonable, when you consider the rotation
        prediction, _ = split_name(prediction)
        if is_correctly_ranked(prediction, rank, ground_truth):
            correctly_ranked += 1

    return correctly_ranked


def get_category(name: str) -> str:
    return name.split("_")[-2]


def get_category_from_list(category: str, selected_categories: List[str], others_name: str = "other") -> str:
    if category not in selected_categories:
        return others_name
    else:
        return category
