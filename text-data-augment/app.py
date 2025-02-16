import os
import glob
import random
import nltk
from nltk.corpus import wordnet
from datasets import load_dataset

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def get_synonyms(word):
    """
    Return a list of synonyms for a given word using NLTK WordNet.
    If no synonyms are found, return an empty list.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Avoid adding the exact same word
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace("_", " ").lower())
    return list(synonyms)


def random_synonym_replacement(sentence, n=1):
    """
    Randomly replace 'n' words in the sentence with one of their synonyms.
    If a word has no synonyms, it remains the same.
    """
    words = sentence.split()
    if len(words) < 2:
        return sentence  # Not enough words to augment

    new_words = words.copy()
    random_word_indices = random.sample(range(len(words)), min(n, len(words)))

    for idx in random_word_indices:
        synonym_list = get_synonyms(words[idx])
        if synonym_list:
            new_words[idx] = random.choice(synonym_list)

    return " ".join(new_words)


def random_insertion(sentence, n=1):
    """
    Randomly insert 'n' words from the sentence into random positions in the same sentence.
    """
    words = sentence.split()
    if len(words) < 2:
        return sentence  # Not enough words to augment

    for _ in range(n):
        word_to_insert = random.choice(words)
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, word_to_insert)

    return " ".join(words)


def random_swap(sentence, n=1):
    """
    Swap two random words in the sentence 'n' times.
    """
    words = sentence.split()
    if len(words) < 2:
        return sentence

    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def random_deletion(sentence, p=0.1):
    """
    Randomly delete each word in the sentence with probability 'p'.
    """
    words = sentence.split()
    if len(words) == 1:
        return sentence  # Avoid deleting the only word

    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    # If everything is deleted, return at least one word
    if len(new_words) == 0:
        new_words.append(random.choice(words))
    return " ".join(new_words)


def augment_sentence(sentence):
    """
    Combine several augmentation methods to produce a new sentence.
    """
    aug_sentence = sentence

    # Synonym Replacement for 1 word
    aug_sentence = random_synonym_replacement(aug_sentence, n=1)

    # Random Insertion of 1 word
    aug_sentence = random_insertion(aug_sentence, n=1)

    # Random Swap of 1 pair of words
    aug_sentence = random_swap(aug_sentence, n=1)

    # Random Deletion with probability 0.1
    aug_sentence = random_deletion(aug_sentence, p=0.1)

    return aug_sentence


def process_and_augment(lines, base_name, output_dir):
    """
    Given a list of text lines, write unaugmented and augmented versions
    to output_dir with names derived from base_name.
    """
    name_only, _ = os.path.splitext(base_name)

    unaug_file = os.path.join(output_dir, f"unaugmented_{name_only}.txt")
    aug_file = os.path.join(output_dir, f"augmented_{name_only}.txt")

    # Write unaugmented data
    with open(unaug_file, "w", encoding="utf-8") as f_unaug:
        for line in lines:
            f_unaug.write(line.strip() + "\n")

    # Write augmented data
    with open(aug_file, "w", encoding="utf-8") as f_aug:
        for line in lines:
            clean_line = line.strip()
            if clean_line:
                augmented_line = augment_sentence(clean_line)
                f_aug.write(augmented_line + "\n")

    return unaug_file, aug_file


def main():
    input_dir = "./input_data"
    output_dir = "./output_data"

    # Create input/output directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Find all .txt files in the input_data folder
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))

    # Prepare a single augmentation description file in output_data
    description_file = os.path.join(output_dir, "augmentation_description.txt")
    description_content = (
        "Augmentation Methods Used:\n"
        "1. Random Synonym Replacement: Replace a random word with a synonym.\n"
        "2. Random Insertion: Insert one random existing word at a random position.\n"
        "3. Random Swap: Swap two random words in the sentence.\n"
        "4. Random Deletion: Each word has a 10% chance to be removed.\n\n"
        "These transformations help increase data variety without collecting new data.\n"
        "---------------------------------------------------------\n"
    )
    processed_files_info = []

    if not input_files:
        # Fallback: Use Hugging Face dataset
        print(
            "No .txt files found in ./input_data. Using IMDB from Hugging Face as a demo dataset."
        )

        # Load a small sample from IMDB (e.g., first 100 lines from the train split)
        dataset = load_dataset("imdb", split="train")
        lines = dataset["text"][:20]

        # Process them as if it were a single "huggingface_imdb.txt"
        base_name = "huggingface_imdb.txt"
        unaug_file, aug_file = process_and_augment(lines, base_name, output_dir)

        info = (
            f"Used fallback Hugging Face dataset: IMDB (train, first 100 samples)\n"
            f"  -> Unaugmented output: {unaug_file}\n"
            f"  -> Augmented output: {aug_file}\n"
        )
        processed_files_info.append(info)

    else:
        # Process each local .txt file found
        for input_file in input_files:
            base_name = os.path.basename(input_file)

            with open(input_file, "r", encoding="utf-8") as f_in:
                lines = list(f_in)

            unaug_file, aug_file = process_and_augment(lines, base_name, output_dir)

            info = (
                f"Processed file: {base_name}\n"
                f"  -> Unaugmented output: {unaug_file}\n"
                f"  -> Augmented output: {aug_file}\n"
            )
            processed_files_info.append(info)

            print(f"Completed augmentation for {base_name}")

    # Write the overall augmentation description
    with open(description_file, "w", encoding="utf-8") as f_desc:
        f_desc.write(description_content)
        f_desc.write("Files Processed:\n")
        for info in processed_files_info:
            f_desc.write(info + "\n")

    print(
        "\nAll files processed. Description of augmentations saved to:",
        description_file,
    )


if __name__ == "__main__":
    main()
