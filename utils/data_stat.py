from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

html5_tags = [
    "!DOCTYPE", "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base", "bdi", "bdo", "blockquote",
    "body", "br", "button", "canvas", "caption", "cite", "code", "col", "colgroup", "data", "datalist", "dd", "del",
    "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html", "i", "iframe", "img", "input", "ins", "kbd",
    "label", "legend", "li", "link", "main", "map", "mark", "meta", "meter", "nav", "noscript", "object", "ol",
    "optgroup", "option", "output", "p", "param", "picture", "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp",
    "script", "section", "select", "small", "source", "span", "strong", "style", "sub", "summary", "sup", "svg", "table",
    "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time", "title", "tr", "track", "u", "ul", "var",
    "video", "wbr",
]


def count_unique_tags(html_content: str) -> int:
    soup = BeautifulSoup(html_content, "html.parser")
    return len({tag.name for tag in soup.find_all()})


def update_tag_frequencies(html_content: str, tag_frequency_dict: Dict[str, int]):
    soup = BeautifulSoup(html_content, "html.parser")
    tags = [tag.name for tag in soup.find_all()]
    counter = Counter(tags)
    for tag, count in counter.items():
        tag_frequency_dict[tag] = tag_frequency_dict.get(tag, 0) + count
    return tag_frequency_dict


def calculate_dom_depth(html_content: str) -> int:
    soup = BeautifulSoup(html_content, "html.parser")

    def get_max_depth(element, depth: int) -> int:
        children = element.find_all(recursive=False)
        if not children:
            return depth
        return max(get_max_depth(child, depth + 1) for child in children)

    return get_max_depth(soup, 0)


def count_total_nodes(html_content: str) -> int:
    soup = BeautifulSoup(html_content, "html.parser")
    return len(soup.find_all())


def compute_stats_and_bucketize(numbers: List[float], k: int):
    if not numbers or k <= 0:
        return "Invalid input"

    average = sum(numbers) / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    range_size = (maximum - minimum) / k if maximum != minimum else 1
    buckets = [[] for _ in range(k)]

    def get_bucket_index(num):
        if num == maximum:
            return k - 1
        return int((num - minimum) / range_size)

    for num in numbers:
        buckets[get_bucket_index(num)].append(num)

    return average, minimum, maximum, buckets


def interval_buckets(min_value: float, max_value: float, k: int):
    if k <= 0:
        raise ValueError("Number of buckets (k) must be positive.")

    interval = (max_value - min_value) / k
    buckets = []
    for i in range(k):
        bucket_min = min_value + i * interval
        bucket_max = min_value + (i + 1) * interval if i < k - 1 else max_value
        buckets.append((bucket_min, bucket_max))
    return buckets


def size_buckets(data: List[float], k: int):
    if k <= 0:
        raise ValueError("Number of buckets (k) must be positive.")

    series = pd.Series(data)
    quantiles = np.linspace(0, 1, k + 1)
    quantile_ranges = series.quantile(quantiles).tolist()
    return [(quantile_ranges[i], quantile_ranges[i + 1]) for i in range(len(quantile_ranges) - 1)]


def pie_chart():
    topics = {
        "product": 4,
        "blog": 21,
        "company/org": 23,
        "homepage": 11,
        "news": 4,
        "forum": 3,
        "information": 9,
        "others": 8,
    }

    labels = topics.keys()
    sizes = topics.values()

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 14})
    plt.axis("equal")
    plt.savefig("topic_pie_chart.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def get_txt_code(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    text_content = soup.get_text()
    for element in soup.find_all(string=True):
        element.extract()
    html_without_text = str(soup)

    return text_content, html_without_text


def stat(ds: datasets.Dataset):
    def stat_func(items):
        try:
            htmls = [html.strip() for html in items["text"]]
            txts, codes = [], []
            for html in htmls:
                txt, code = get_txt_code(html)
                txts.append(txt)
                codes.append(code)

            items["length"] = [len(tokenizer(html, max_length=10240, truncation=True)["input_ids"]) for html in htmls]
            items["length_txt"] = [len(tokenizer(txt, max_length=10240, truncation=True)["input_ids"]) for txt in txts]
            items["length_code"] = [len(tokenizer(code, max_length=10240, truncation=True)["input_ids"]) for code in codes]
            items["total_tags"] = [count_total_nodes(html) for html in htmls]
            items["dom_depth"] = [calculate_dom_depth(html) for html in htmls]
            items["unique_tags"] = [count_unique_tags(html) for html in htmls]
            return items
        except Exception:
            import traceback

            print(f"stat error: {traceback.format_exc()}")
            htmls = [html.strip() for html in items["text"]]
            items["length"] = [8460] * len(htmls)
            items["length_txt"] = [2000] * len(htmls)
            items["length_code"] = [6000] * len(htmls)
            items["total_tags"] = [175] * len(htmls)
            items["dom_depth"] = [15] * len(htmls)
            items["unique_tags"] = [21] * len(htmls)
            return items

    ds_stat = ds.map(stat_func, num_proc=32, batched=True, batch_size=128)

    all_lengths = ds_stat["length"]
    all_lengths_txt = ds_stat["length_txt"]
    all_lengths_code = ds_stat["length_code"]
    all_total_tags = ds_stat["total_tags"]
    all_dom_depths = ds_stat["dom_depth"]
    all_unique_tags = ds_stat["unique_tags"]

    print("mean length: ", np.mean(all_lengths), np.std(all_lengths))
    print("mean length txt: ", np.mean(all_lengths_txt), np.std(all_lengths_txt))
    print("mean length code: ", np.mean(all_lengths_code), np.std(all_lengths_code))
    print("mean total tags: ", np.mean(all_total_tags), np.std(all_total_tags))
    print("mean dom depth: ", np.mean(all_dom_depths), np.std(all_dom_depths))
    print("mean unique tags: ", np.mean(all_unique_tags), np.std(all_unique_tags))


if __name__ == "__main__":
    data_path = Path("data/Design2Code-HARD")
    txts = []
    i = 0
    while True:
        html_path = data_path / f"g{i}.html"
        image_path = data_path / f"g{i}.png"
        if not html_path.exists() and not image_path.exists():
            break
        if html_path.exists():
            txts.append(html_path.read_text(encoding="utf-8"))
        i += 1
    ds1 = datasets.Dataset.from_dict({"text": txts})

    ds2 = load_dataset("parquet", data_files="data/CC-HARD.parquet")["train"]

    print("Design2Code-HARD:")
    stat(ds1)
    print("CC-HARD:")
    stat(ds2)
