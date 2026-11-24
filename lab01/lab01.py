import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

# Части речи, допустимые на первом и втором месте пары.
ALLOWED_POS = frozenset({"NOUN", "ADJF"})

# Граммемы словарных сокращений и "заглушек": "в." = век, "и." = прочее и т.п.
FORBIDDEN_GRAMMEMES = frozenset({"Abbr", "Fixd", "Init"})

# Собственные имена: разбор принимается только для слова с заглавной буквы, иначе "машин" (от "Маша") конкурирует с притяжательным прилагательным.
PROPER_GRAMMEMES = frozenset({"Surn", "Name", "Patr", "Geox"})

# Падежи, которые не являются настоящими падежами: звательный (voct) и прочие словарные "падежи" OpenCorpora, для согласования не годятся.
NON_AGREEMENT_CASES = frozenset({None, "voct"})

# Вторичные падежи - варианты основных, а не отдельные падежи. Если их не нормализовать, теряются корректные пары.
SECONDARY_CASES = {"loc2": "loct", "gen2": "gent", "acc2": "accs"}


def normalize_case(case: str | None) -> str | None:
    """
    Свести вторичный падеж (loc2/gen2/acc2) к основному.
    """
    if case is None:
        return None
    return SECONDARY_CASES.get(case, case)


@dataclass(frozen=True)
class Token:
    """
    Словоформа с одним допустимым морфологическим разбором.
    """

    word: str
    lemma: str
    pos: str
    gender: str | None
    number: str | None
    case: str | None

    def describe(self) -> str:
        """
        POS и граммемы, участвующие в согласовании.
        """
        parts = [self.pos]
        for value in (self.gender, self.number, self.case):
            if value:
                parts.append(value)
        return ",".join(parts)


@dataclass(frozen=True)
class Pair:
    """
    Найденная пара: исходные словоформы и результат в виде лемм.
    """

    first: Token
    second: Token

    @property
    def lemmas(self) -> tuple[str, str]:
        return self.first.lemma, self.second.lemma

    @property
    def words(self) -> tuple[str, str]:
        return self.first.word, self.second.word

    @property
    def combo(self) -> str:
        return f"{self.first.pos}+{self.second.pos}"

    def render(self, verbose: bool = False) -> str:
        lemma_pair = " ".join(self.lemmas)
        if not verbose:
            return lemma_pair
        word_pair = " ".join(self.words)
        detail = f"[{self.combo}] {self.first.describe()} | {self.second.describe()}"
        return f"{lemma_pair:<34} <- {word_pair:<28} {detail}"


def candidate_tokens(word: str, morph: pymorphy3.MorphAnalyzer) -> list[Token]:
    """
    Все допустимые разборы слова, а не только самый вероятный.

    Берем все разборы, потому что pymorphy3 не учитывает контекст.
    """
    is_capitalized = word[:1].isupper()
    result: list[Token] = []

    for parsed in morph.parse(word):
        tag = parsed.tag
        grammemes = set(tag.grammemes)

        if tag.POS not in ALLOWED_POS:
            continue
        if grammemes & FORBIDDEN_GRAMMEMES:
            continue
        if (grammemes & PROPER_GRAMMEMES) and not is_capitalized:
            continue

        result.append(
            Token(
                word=word,
                lemma=parsed.normal_form,
                pos=tag.POS,
                gender=getattr(tag, "gender", None),
                number=getattr(tag, "number", None),
                case=normalize_case(getattr(tag, "case", None)),
            )
        )

    return result


def agree(first: Token, second: Token) -> bool:
    """
    Согласование по роду, числу и падежу.

    - число задано у обоих и совпадает;
    - падеж задан у обоих и совпадает;
    - при множественном числе род в русском языке не выражен, поэтому он не сравнивается;
    - в единственном числе род должен совпадать либо отсутствовать у одной из сторон (например, несклоняемые/собирательные существительные).
    """
    if first.number is None or second.number is None:
        return False
    if first.number != second.number:
        return False

    if first.case in NON_AGREEMENT_CASES or second.case in NON_AGREEMENT_CASES:
        return False
    if first.case != second.case:
        return False

    if first.number == "plur":
        return True

    if first.gender is None or second.gender is None:
        return True
    return first.gender == second.gender


def ensure_nltk_data() -> None:
    """
    Догрузить модели NLTK при первом запуске.
    """
    for resource, package in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def split_sentences(text: str) -> list[str]:
    """
    Сегментация текста на предложения.
    """
    return sent_tokenize(text)


def token_spans(sentence: str) -> list[list[str]]:
    """
    Группы подряд идущих слов, не разделенных пунктуацией.
    """
    groups: list[list[str]] = []
    current: list[str] = []

    for token in word_tokenize(sentence):
        if any(char.isalpha() for char in token):
            current.append(token)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    return groups


def find_pairs(text: str, morph: pymorphy3.MorphAnalyzer) -> list[Pair]:
    """
    Все пары согласованных существительных/прилагательных в тексте.
    """
    pairs: list[Pair] = []

    for sentence in split_sentences(text):
        for group in token_spans(sentence):
            analyses = [candidate_tokens(word, morph) for word in group]

            for index in range(len(group) - 1):
                # Дедупликация по паре лемм: падежная омонимия дает один и тот же результат дважды.
                seen: set[tuple[str, str]] = set()

                for first in analyses[index]:
                    for second in analyses[index + 1]:
                        if not agree(first, second):
                            continue
                        pair = Pair(first, second)
                        if pair.lemmas in seen:
                            continue
                        seen.add(pair.lemmas)
                        pairs.append(pair)

    return pairs


def unique_lemma_pairs(pairs: list[Pair]) -> list[tuple[str, str]]:
    """
    Уникальные пары лемм в порядке первого появления.
    """
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair.lemmas not in unique:
            unique.append(pair.lemmas)
    return unique


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    default_input = Path(__file__).with_name("text.txt")

    parser = argparse.ArgumentParser(
        description="Пары соседних существительных/прилагательных, "
        "согласованных по роду, числу и падежу (вывод в виде лемм)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"файл с текстом (по умолчанию {default_input.name})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="показать исходные словоформы и грамматические признаки",
    )
    args = parser.parse_args(argv)

    try:
        text = read_text(args.input)
    except OSError as error:
        print(f"Не удалось прочитать файл: {error}", file=sys.stderr)
        return 1

    ensure_nltk_data()

    morph = pymorphy3.MorphAnalyzer()
    pairs = find_pairs(text, morph)

    if not pairs:
        print("Пары не найдены.")
        return 0

    for pair in pairs:
        print(pair.render(verbose=args.verbose))

    unique = unique_lemma_pairs(pairs)
    print(f"\nВсего пар: {len(pairs)}, уникальных пар лемм: {len(unique)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
