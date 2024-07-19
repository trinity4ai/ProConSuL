test_names = []  # list to save the order
percentages = {}
failures_sum = {}
items_sum = {}


def register(test_name: str) -> None:
    if test_name in test_names:
        raise RuntimeError(f"Test '{test_name}' is already registered")
    test_names.append(test_name)
    percentages[test_name] = []
    failures_sum[test_name] = 0
    items_sum[test_name] = 0


def add_result(test_name: str, failures: float, all_items: float) -> None:
    if test_name not in test_names:
        register(test_name)
    percentages[test_name].append(round(failures / all_items * 10000) / 10000)
    failures_sum[test_name] += failures
    items_sum[test_name] += all_items


def output_statistics(test_name: str) -> str:
    return (f'Total statistics for {test_name}:\n'
            f' MIN: {min(percentages[test_name])}\n'
            f' MAX: {max(percentages[test_name])}\n'
            f' Among all verified items in all run tests:'
            f' {round(failures_sum[test_name] / items_sum[test_name] * 10000) / 10000}\n\n')


def all_statistics() -> str:
    result = ""
    for test_name in test_names:
        result += output_statistics(test_name)
    return result


def average_score(test_name: str) -> float:
    return round(failures_sum[test_name] / items_sum[test_name], 2)


def all_scores_to_csv() -> str:
    result = ""
    for test_name in test_names:
        result += test_name
        for percent in percentages[test_name]:
            result += ", " + str(percent)
        result += f", Average:, {average_score(test_name)}\n"
    return result
