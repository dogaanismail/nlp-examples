import pathlib
import spacy
from spacy import Language
from collections import Counter
from spacy.tokenizer import Tokenizer

# Ensure the model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print(nlp)  # Just to confirm that 'nlp' is correctly initialized

# %% Introduction

introduction_text = (
    "This tutorial is about Natural Language Processing in spaCy."
)
introduction_doc = nlp(introduction_text)

# Extract tokens for the given doc
print([token.text for token in introduction_doc])

# %% Reading text from a file instead

file_name = "introduction.txt"
introduction_file_text = pathlib.Path(file_name).read_text()
introduction_file_doc = nlp(introduction_file_text)

# %% Extracting tokens and reading on one line

# Extract tokens for the given doc
print([token.text for token in nlp(pathlib.Path(file_name).read_text())])

# %% Extracting sentences

about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)
about_doc = nlp(about_text)
sentences = list(about_doc.sents)
len(sentences)

for sentence in sentences:
    print(f"{sentence[:5]}...")

print(type(sentences))


# %% Customizing sentence boundaries

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    """Adds support to use `...` as the delimiter for sentence detection"""
    for token in doc[:-1]:
        if token.text == "...":
            doc[token.i + 1].is_sent_start = True
    return doc


ellipsis_text = (
    "Gus, can you, ... never mind, I forgot"
    " what I was saying. So, do you think"
    " we should ..."
)

# Load a new model instance
custom_nlp = spacy.load("en_core_web_sm")
custom_nlp.add_pipe("set_custom_boundaries", before="parser")
custom_ellipsis_doc = custom_nlp(ellipsis_text)
custom_ellipsis_sentences = list(custom_ellipsis_doc.sents)
for sentence in custom_ellipsis_sentences:
    print(sentence)

ellipsis_doc = nlp(ellipsis_text)
ellipsis_sentences = list(ellipsis_doc.sents)
for sentence in ellipsis_sentences:
    print(sentence)

# %% Each token has its index position in the original text

for token in about_doc:
    print(token, token.idx)

# %% Each token has various attributes you can use

print(
    f"{'Text with Whitespace':22}"
    f"{'Is Alphanum?':15}"
    f"{'Is Punctuation?':18}"
    f"{'Is Stop Word?'}"
)

for token in about_doc:
    print(
        f"{str(token.text_with_ws):22}"
        f"{str(token.is_alpha):15}"
        f"{str(token.is_punct):18}"
        f"{str(token.is_stop)}"
    )

# %% Customizing the tokenizer to add a custom infix

custom_about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London@based London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)

print([token.text for token in nlp(custom_about_text)[8:15]])

custom_nlp = spacy.load("en_core_web_sm")
prefix_re = spacy.util.compile_prefix_regex(custom_nlp.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(custom_nlp.Defaults.suffixes)

custom_infixes = [r"@"]
# A more complete regex pattern:
# custom_infixes = [r"(?<=[a-zA-Z_])@(?=[a-zA-Z_])"]

infix_re = spacy.util.compile_infix_regex(
    list(custom_nlp.Defaults.infixes) + custom_infixes
)

custom_nlp.tokenizer = Tokenizer(
    nlp.vocab,
    prefix_search=prefix_re.search,
    suffix_search=suffix_re.search,
    infix_finditer=infix_re.finditer,
    token_match=None,
)

custom_tokenizer_about_doc = custom_nlp(custom_about_text)

print([token.text for token in custom_tokenizer_about_doc[8:15]])

# %% The language model includes stop words

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
len(spacy_stopwords)

for stop_word in list(spacy_stopwords)[:15]:
    print(stop_word)

# %% Words in `about_doc`` that aren't stop words

for token in about_doc:
    if not token.is_stop:
        print(token)

print(list(filter(lambda t: not t.is_stop, about_doc)))

print([token for token in about_doc if not token.is_stop])

about_no_stopword_doc = [token for token in about_doc if not token.is_stop]
print(about_no_stopword_doc)

# %% Lemmas and lemmatization

conference_help_text = (
    "Gus is helping organize a developer"
    " conference on Applications of Natural Language"
    " Processing. He keeps organizing local Python meetups"
    " and several internal talks at his workplace."
)
conference_help_doc = nlp(conference_help_text)
for token in conference_help_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")

# %% Making use of stop words to count words that aren't stop words

complete_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors."
)

complete_doc = nlp(complete_text)
# Remove stop words and punctuation symbols
words = [
    token.text
    for token in complete_doc
    if not token.is_stop and not token.is_punct
]
word_freq = Counter(words)
# 5 commonly occurring words with their frequencies
common_words = word_freq.most_common(5)
print(common_words)

# Unique words
unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
print(unique_words)

# %% What the same count would look like with stop words included

words_all = [token.text for token in complete_doc if not token.is_punct]
print(
    Counter(
        [token.text for token in complete_doc if not token.is_punct]
    ).most_common(5)
)

# %% Part-of-speech tagging

for token in about_doc[:5]:
    print(
        f"""
        TOKEN: {str(token)}
        =====
        TAG: {str(token.tag_):10} POS: {token.pos_}
        EXPLANATION: {spacy.explain(token.tag_)}"""
    )
