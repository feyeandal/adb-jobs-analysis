

#function to identify non alpha numeric characters

def get_special_chars(text_column):
    
    #converting to a single string
    text = ' '.join(text_column)
    
    # get a list of unique characters
    text_char = list(set(text))
    
    # get a list removing alpha numeric
    text_char_sp = [char for char in text_char if not(char.isalnum())]
    
    return text_char_sp

# execute the function on the i2t list to get a list of special characters
special = get_special_chars(i2t)

# function to remove special characters
def strip_special_chars(text, schar_list, char_keep):
    """
    Strips the unwanted special characters from a given list of job descriptions

    Parameters:
    ----------
    text : list of job descriptions
    schar_list : relevant special character list
    char_keep : the special characters to be retained

    Returns:
    -------
    The list of job descriptions stripped of unwanted special characters
    """
    char_set = set([c for c in schar_list if c not in char_keep])
    
    # i2t_stripped -> stripped of special chars
    text_stripped = [''.join([c for c in item if c not in char_set]) for item in text]
    
    return text_stripped

# define characters you want to retain
char_keep = [' ', '#', '+', '\n', '/']

# execute the function and obtain ocr output stripped of special characters
stripped = strip_special_chars(i2t, special, char_keep)

df["cleaned_text"] = pd.Series(stripped)

#######################################################################

# Dictionary LookUp

#######################################################################

def accuracy_calculator(string):
    
    valid_count = 0
#     invalid = 0
    
    split_list = string.split()
    for word in split_list:
        if dict.check(word) == True:
            valid_count += 1
#         else:
#             invalid.append(word)
    return (valid_count/len(split_list))
    
    