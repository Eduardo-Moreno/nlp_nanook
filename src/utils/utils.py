import re
import seaborn as sns
import matplotlib.pyplot as plt
from unidecode import unidecode
from collections import Counter

#################### Variables ####################

symbols = [
    '¿','?','~','`','!','¡','@','#','$','%','^',
    '*','(',')','_','-','+','=','{','}','[',
    ']','\\',':',';','<','>','/', '.', ',','&','\n'
    ,'\r','\t', '|', '“', '"', '-', '”', '©', '-', '—',
    '…', ';', '‘','’'
]
patron_url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

#################### Functions ####################

def formato_abreviado(valor, _) -> str:
    """ Función para formatear los números en forma abreviada (por ejemplo, 140000 -> 140k)

    Args:
        valor (_type_): _description_
        _ (_type_): _description_

    Returns:
        _type_: _description_
    """

    if valor >= 1e6:
        return f'{int(valor/1e6)}M'
    elif valor >= 1e3:
        return f'{int(valor/1e3)}k'
    else:
        return str(int(valor))

def remove_symbols(text:str) -> str:
    """ Remove punctuation symbols from string

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """

    res = text
    for char in symbols:
        res = res.replace(char, '')
    return res

def clean_text(text:str) -> str:
    """ Clean text

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """

    # Ommit URLs
    res = patron_url.sub('', text)
    # Text to lowercase
    res = res.lower()
    # Ommit accents
    res = unidecode(res)
    # Ommit symbols
    res = remove_symbols(res)
    return res

def generate_N_grams(text, ngram=1, stopwords=None) -> list:
    """Generate N grams for single text

    Args:
        text (_type_): _description_
        ngram (int, optional): _description_. Defaults to 1.
        stopwords (_type_, optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """

    if stopwords == None:
        words=[word for word in text.split(" ")]
    else:
        words=[word for word in text.split(" ") if word not in stopwords]
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def get_corpus_N_gram(list_text:list, stop_words:list, ngram:int, show_plot=False) -> list:
    """Get corpus and plot most frequently ngrams

    Args:
        list_text (list): _description_
        stop_words (list): _description_
        ngram (int): _description_
        show_plot (bool, optional): _description_. Defaults to False.

    Returns:
        list: _description_
    """

    # Compute corpus
    corpus = []
    for x in list_text:
        corpus += generate_N_grams(x, stopwords=stop_words, ngram=ngram)
    print(f"Length of corpus: {len(corpus)}")
    # Counter corpus
    mostCommon = Counter(corpus).most_common(15)
    words = []
    freq = []
    for word, count in mostCommon:
        words.append(word)
        freq.append(count)
    # Plot
    if show_plot:
        sns.barplot(x=freq, y=words)
        plt.title(f'Top 15 Most Frequently Occuring {ngram}-gram')
        plt.show()
    
    return corpus

def remove_stop_words(text, stop_words):
    """ Remove stop words from the input text.
    """

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_stop_words_from_dataframe(df, columns, stop_words_file):
    """ Remove stop words from all text columns in the DataFrame.
    """
    
    # Load stop words from file
    with open(stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())
    
    # Apply remove_stop_words function to all text columns
    for column in columns:
        aux = df[column].apply(lambda x: remove_stop_words(x, stop_words))
        df[column + '_stpWrd'] = aux
    return df