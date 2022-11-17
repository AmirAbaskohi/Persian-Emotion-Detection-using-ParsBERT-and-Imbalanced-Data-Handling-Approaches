from bert_spell_checker import BertSpellChecker
import tokenizer
import emojis
import requests

def extract_misspelled_words(text):
    tokenizer_object = tokenizer.Tokenizer()
    tokenized_originial = tokenizer_object(text)

    spell_checker = BertSpellChecker(max_edit_distance_dictionary=2, prefix_length=7,
                          bert_model='HooshvareLab/bert-base-parsbert-uncased')
    spell_checker.set_dictionaries()
    incorrect_words, masked_text, incorrect_words_position = spell_checker.get_misspelled_words_and_masked_text(text)
    corrected_text = spell_checker.get_bert_suggestion_for_each_mask(text, incorrect_words_position, num_suggestions=10)

    tokenized_corrected = tokenizer_object(corrected_text)
    misspelled_words = [i for i in tokenized_originial + tokenized_corrected if i not in tokenized_originial or i not in tokenized_corrected]
    return f"</s>{','.join(misspelled_words)}</s>"

def extract_emojies(text):
    emojies = list(emojis.get(text))
    if ":)" in text or "(:" in text:
        emojies.append(":)")

    if ":|" in text or "|:" in text:
        emojies.append(":|")

    if ":(" in text or "):" in text:
        emojies.append(":(")

    if "<3" in text:
        emojies.append("<3")

    return f"</s>{','.join(emojies)}</s>"

def extract_hashtags(text):
    hashtag_list = []
    for word in text.split():
        if word[0] == '#':
            hashtag_list.append(word[1:])
    return f"</s>{','.join(hashtag_list)}</s>"

def query(payload, API_URL, headers):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def extract_POS_tags(text):

    API_URL = "https://api-inference.huggingface.co/models/wietsedv/xlm-roberta-base-ft-udpos28-fa"
    headers = {"Authorization": "Bearer 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"}
        
    output = query({
        "inputs": text,
    }, API_URL, headers)

    POS_tags = [entity['entity_group'] for entity in output]
    return f"</s>{','.join(POS_tags)}</s>"