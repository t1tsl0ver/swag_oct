r"""Build a self-contained, interactive overview of the competition CSV files.

Run from the repository root:
    .\.venv\Scripts\python.exe src\swag_oct\visualization\overview.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from math import ceil, isfinite
from pathlib import Path
from typing import Callable

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[3]
MISSING = "<пропуск>"
SOURCE_URL = "https://www.kaggle.com/competitions/playground-series-s6e10"
TRAIN_COLOR = "#2563eb"
TEST_COLOR = "#f59e0b"
RATE_COLOR = "#0f766e"


@dataclass
class FeatureCounts:
    train: Counter[str] = field(default_factory=Counter)
    test: Counter[str] = field(default_factory=Counter)
    positive: Counter[str] = field(default_factory=Counter)


@dataclass
class DatasetCounts:
    features: list[str]
    values: dict[str, FeatureCounts]
    train_rows: int = 0
    test_rows: int = 0
    positives: int = 0
    class_travel: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )


def read_counts(train_path: Path, test_path: Path) -> DatasetCounts:
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Нужны data/train.csv и data/test.csv")

    with train_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "satisfaction" not in reader.fieldnames:
            raise ValueError("В train.csv нет столбца satisfaction")
        features = [name for name in reader.fieldnames if name not in ("id", "satisfaction")]
        result = DatasetCounts(features, {name: FeatureCounts() for name in features})
        for row in reader:
            label = row["satisfaction"].strip().lower()
            if label not in ("true", "false", "1", "0"):
                raise ValueError(f"Неизвестная метка satisfaction: {label!r}")
            positive = label in ("true", "1")
            result.train_rows += 1
            result.positives += positive
            for name in features:
                value = row[name].strip() or MISSING
                result.values[name].train[value] += 1
                result.values[name].positive[value] += positive
            if "Class" in row and "Type of Travel" in row:
                pair = (row["Class"].strip(), row["Type of Travel"].strip())
                result.class_travel[pair][0] += 1
                result.class_travel[pair][1] += positive

    with test_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or set(reader.fieldnames) != set(features) | {"id"}:
            raise ValueError("Набор признаков test.csv отличается от train.csv")
        for row in reader:
            result.test_rows += 1
            for name in features:
                result.values[name].test[row[name].strip() or MISSING] += 1

    if not result.train_rows or not result.test_rows:
        raise ValueError("Один из файлов данных пуст")
    return result


def _numeric(counter: Counter[str]) -> bool:
    try:
        return all(isfinite(float(value)) for value in counter if value != MISSING)
    except ValueError:
        return False


def _weighted_quantile(counter: Counter[str], quantile: float) -> float:
    ordered = sorted((float(value), count) for value, count in counter.items() if value != MISSING)
    rank = ceil(sum(count for _, count in ordered) * quantile)
    cumulative = 0
    for value, count in ordered:
        cumulative += count
        if cumulative >= rank:
            return value
    return ordered[-1][0]


def _aggregate(counter: Counter[str], classify: Callable[[str], str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for value, count in counter.items():
        result[classify(value)] += count
    return result


def _feature_bins(
    counts: FeatureCounts,
) -> tuple[list[str], Counter[str], Counter[str], Counter[str], list[str], Counter[str], Counter[str]]:
    """Return distribution and target-rate bins without materializing raw rows."""
    raw_values = set(counts.train) | set(counts.test)
    continuous = _numeric(counts.train) and len(raw_values - {MISSING}) > 20

    if continuous:
        minimum = min(float(value) for value in raw_values if value != MISSING)
        cutoff = _weighted_quantile(counts.train, 0.99)
        if cutoff <= minimum:
            cutoff = max(float(value) for value in raw_values if value != MISSING)
        width = (cutoff - minimum) / 24 if cutoff > minimum else 1.0
        labels = [f"{minimum + i * width:g}–{minimum + (i + 1) * width:g}" for i in range(24)]
        overflow = f">{cutoff:g}"
        if any(float(value) > cutoff for value in raw_values if value != MISSING):
            labels.append(overflow)

        def distribution_bin(value: str) -> str:
            if value == MISSING:
                return MISSING
            number = float(value)
            if number > cutoff:
                return overflow
            index = min(max(int((number - minimum) / width), 0), 23)
            return labels[index]

        boundaries = sorted({_weighted_quantile(counts.train, step / 10) for step in range(1, 10)})
        target_labels = []
        for index, bound in enumerate(boundaries):
            target_labels.append(f"≤{bound:g}" if index == 0 else f"{boundaries[index - 1]:g}–{bound:g}")
        target_labels.append(f">{boundaries[-1]:g}" if boundaries else "все значения")

        def target_bin(value: str) -> str:
            if value == MISSING:
                return MISSING
            number = float(value)
            for index, bound in enumerate(boundaries):
                if number <= bound:
                    return target_labels[index]
            return target_labels[-1]

    else:
        if _numeric(counts.train):
            labels = sorted(raw_values - {MISSING}, key=float)
        else:
            labels = sorted(raw_values - {MISSING}, key=lambda value: (-counts.train[value], value))
        if MISSING in raw_values:
            labels.append(MISSING)
        target_labels = labels

        def distribution_bin(value: str) -> str:
            return value

        target_bin = distribution_bin

    train_distribution = _aggregate(counts.train, distribution_bin)
    test_distribution = _aggregate(counts.test, distribution_bin)
    target_total = _aggregate(counts.train, target_bin)
    target_positive = _aggregate(counts.positive, target_bin)
    if MISSING in train_distribution or MISSING in test_distribution:
        labels = [*labels, MISSING] if MISSING not in labels else labels
    if MISSING in target_total and MISSING not in target_labels:
        target_labels = [*target_labels, MISSING]
    return labels, train_distribution, test_distribution, counts.train, target_labels, target_total, target_positive


def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=45, r=20, t=55, b=85),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#172033"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.13),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e7ebf2", zeroline=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return fig


def _chart_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def _feature_charts(name: str, counts: FeatureCounts, train_rows: int, test_rows: int, baseline: float) -> tuple[str, str, float]:
    labels, train_hist, test_hist, _, target_labels, target_total, target_positive = _feature_bins(counts)
    train_share = [train_hist[label] / train_rows for label in labels]
    test_share = [test_hist[label] / test_rows for label in labels]
    shift = sum(abs(train - test) for train, test in zip(train_share, test_share)) / 2

    distribution = go.Figure()
    distribution.add_bar(x=labels, y=train_share, name="Train", marker_color=TRAIN_COLOR)
    distribution.add_bar(x=labels, y=test_share, name="Test", marker_color=TEST_COLOR)
    distribution.update_layout(title="Распределение значений", barmode="group", yaxis_tickformat=".1%")
    distribution.update_traces(hovertemplate="%{x}<br>Доля: %{y:.2%}<extra>%{fullData.name}</extra>")

    present = [label for label in target_labels if target_total[label]]
    rates = [target_positive[label] / target_total[label] for label in present]
    sizes = [target_total[label] for label in present]
    response = go.Figure()
    response.add_scatter(
        x=present,
        y=rates,
        customdata=sizes,
        mode="lines+markers",
        marker=dict(color=RATE_COLOR, size=9),
        line=dict(color=RATE_COLOR, width=3),
        name="Доля довольных",
        hovertemplate="%{x}<br>Доля довольных: %{y:.2%}<br>Строк train: %{customdata:,}<extra></extra>",
    )
    response.add_hline(y=baseline, line_dash="dot", line_color="#9ca3af", annotation_text="Среднее по train")
    response.update_layout(title="Связь с satisfaction", yaxis_tickformat=".0%", yaxis_range=[0, 1])
    return _chart_html(_style(distribution)), _chart_html(_style(response)), shift


def _overview_charts(dataset: DatasetCounts, shifts: dict[str, float]) -> tuple[str, str, str, str]:
    balance = go.Figure(go.Bar(
        x=["Недовольны", "Довольны"],
        y=[dataset.train_rows - dataset.positives, dataset.positives],
        marker_color=["#64748b", RATE_COLOR],
        text=[dataset.train_rows - dataset.positives, dataset.positives],
        textposition="outside",
    ))
    balance.update_layout(title="Целевая переменная в train", showlegend=False)

    missing_names = [name for name in dataset.features if dataset.values[name].train[MISSING] or dataset.values[name].test[MISSING]]
    missing = go.Figure()
    missing.add_bar(x=missing_names, y=[dataset.values[name].train[MISSING] / dataset.train_rows for name in missing_names], name="Train", marker_color=TRAIN_COLOR)
    missing.add_bar(x=missing_names, y=[dataset.values[name].test[MISSING] / dataset.test_rows for name in missing_names], name="Test", marker_color=TEST_COLOR)
    missing.update_layout(title="Пропуски по признакам", barmode="group", yaxis_tickformat=".2%")

    ordered = sorted(shifts, key=shifts.get)
    drift = go.Figure(go.Bar(x=[shifts[name] for name in ordered], y=ordered, orientation="h", marker_color="#7c3aed"))
    drift.update_layout(title="Различие распределений train и test по отдельным признакам", xaxis_title="TV distance: 0 — совпадение")

    classes = sorted({key[0] for key in dataset.class_travel})
    travels = sorted({key[1] for key in dataset.class_travel})
    rate_matrix = []
    hover_matrix = []
    for travel in travels:
        rate_row = []
        hover_row = []
        for group in classes:
            total, positive = dataset.class_travel[(group, travel)]
            rate_row.append(positive / total if total else None)
            hover_row.append(total)
        rate_matrix.append(rate_row)
        hover_matrix.append(hover_row)
    interaction = go.Figure(go.Heatmap(
        x=classes, y=travels, z=rate_matrix, customdata=hover_matrix,
        colorscale="Teal", zmin=0, zmax=1, colorbar=dict(tickformat=".0%"),
        hovertemplate="Class: %{x}<br>Type of Travel: %{y}<br>Доля довольных: %{z:.1%}<br>Строк: %{customdata:,}<extra></extra>",
    ))
    interaction.update_layout(title="Доля довольных: класс × цель поездки")

    return (
        _chart_html(_style(balance)),
        _chart_html(_style(missing)),
        _chart_html(_style(drift, 600)),
        _chart_html(_style(interaction, 420)),
    )


def build_report(dataset: DatasetCounts) -> str:
    baseline = dataset.positives / dataset.train_rows
    feature_sections = []
    shifts = {}
    options = []
    for index, name in enumerate(dataset.features):
        counts = dataset.values[name]
        distribution, response, shift = _feature_charts(name, counts, dataset.train_rows, dataset.test_rows, baseline)
        shifts[name] = shift
        options.append(f'<option value="{index}">{escape(name)}</option>')
        train_missing = counts.train[MISSING] / dataset.train_rows
        test_missing = counts.test[MISSING] / dataset.test_rows
        feature_sections.append(
            f'<section class="feature-panel" data-feature="{index}" style="display:{"block" if index == 0 else "none"}">'
            f'<h3>{escape(name)}</h3><p class="meta">Уникальных значений в train: {len(counts.train) - (MISSING in counts.train):,} '
            f'· Пропуски train/test: {train_missing:.3%} / {test_missing:.3%} '
            f'· Различие распределений (TV): {shift:.3%}</p>'
            f'<div class="chart-grid">{distribution}{response}</div></section>'
        )

    balance, missing, drift, interaction = _overview_charts(dataset, shifts)
    top_shift = sorted(shifts, key=shifts.get, reverse=True)[:3]
    top_shift_text = ", ".join(f"{escape(name)} ({shifts[name]:.2%})" for name in top_shift)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обзор данных — Playground S6E10</title>
<style>
body {{margin:0;background:#f3f6fb;color:#172033;font:16px/1.55 Arial,sans-serif}}
main {{max-width:1250px;margin:0 auto;padding:32px 24px 60px}}
h1 {{font-size:32px;margin:0 0 8px}} h2 {{font-size:23px;margin:36px 0 14px}} h3 {{font-size:20px;margin:0}}
.lead {{color:#475569;max-width:950px}} .cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:25px 0}}
.card,.panel {{background:white;border:1px solid #e3e9f1;border-radius:14px;box-shadow:0 3px 14px #1c35510a}}
.card {{padding:17px 20px}} .card strong {{font-size:26px;display:block;color:#0f766e}} .card span,.meta {{color:#64748b}}
.panel {{padding:18px;margin:16px 0}} .chart-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px}}
select {{font:inherit;padding:10px 14px;border:1px solid #cad4e2;border-radius:8px;max-width:100%;background:white}}
.note {{border-left:4px solid #f59e0b;padding:13px 18px;background:#fffbeb;border-radius:5px;margin:22px 0}}
a {{color:#1d4ed8}} @media(max-width:700px) {{main {{padding:20px 12px}} .chart-grid {{grid-template-columns:1fr}}}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>Данные соревнования: первый обзор</h1>
<p class="lead">Интерактивный отчёт по локальным CSV из <a href="{SOURCE_URL}">Kaggle Playground S6E10</a>. Создан {escape(generated)}. Наведи курсор на столбцы и точки, чтобы увидеть значения.</p>
<div class="cards">
<div class="card"><strong>{dataset.train_rows:,}</strong><span>строк train</span></div>
<div class="card"><strong>{dataset.test_rows:,}</strong><span>строк test</span></div>
<div class="card"><strong>{len(dataset.features)}</strong><span>признак</span></div>
<div class="card"><strong>{baseline:.2%}</strong><span>довольных в train</span></div>
</div>
<div class="note">Графики показывают свойства файлов, но сами по себе не доказывают, были ли отдельные строки созданы синтетически. Большое различие train/test по одному признаку ещё не означает различие по всем признакам вместе.</div>
<h2>Общая картина</h2><div class="chart-grid"><div class="panel">{balance}</div><div class="panel">{missing}</div></div>
<div class="panel">{interaction}</div>
<div class="panel">{drift}<p class="meta">TV = ½ · сумма |доля train − доля test| по показанным корзинам. Это описательная мера для одного признака, не результат классификатора и не доказательство утечки. Наибольшие значения: {top_shift_text}.</p></div>
<h2>Исследовать отдельный признак</h2>
<label for="feature-picker">Признак:</label> <select id="feature-picker">{''.join(options)}</select>
<p class="meta">Числовые признаки с большим числом значений сгруппированы: распределение — в интервалы одинаковой ширины до 99-го процентиля, доля довольных — примерно по децилям train. Пропуски показываются отдельно.</p>
<div class="panel">{''.join(feature_sections)}</div>
<script>
const picker = document.getElementById('feature-picker');
picker.addEventListener('change', () => {{
  document.querySelectorAll('.feature-panel').forEach(panel => {{ panel.style.display = panel.dataset.feature === picker.value ? 'block' : 'none'; }});
  const selected = document.querySelector('.feature-panel[data-feature="' + picker.value + '"]');
  requestAnimationFrame(() => selected.querySelectorAll('.js-plotly-plot').forEach(plot => Plotly.Plots.resize(plot)));
}});
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "data_overview.html")
    args = parser.parse_args()
    dataset = read_counts(args.train, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(dataset), encoding="utf-8")
    print(f"Отчёт сохранён: {args.output}")
    print(f"Train: {dataset.train_rows:,}; test: {dataset.test_rows:,}; признаков: {len(dataset.features)}")


if __name__ == "__main__":
    main()
