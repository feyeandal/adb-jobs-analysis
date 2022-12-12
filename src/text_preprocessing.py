
def get_special_chars(texts):
    """identify special characters that need to be removed before evaluation from extracted job descriptions
        args:
            texts [List(str)]: The texts extracted from images
    """
    
    #converting to a single string
    text = ' '.join(texts)
    
    # get a list of unique characters
    text_char = list(set(text))
    
    # get a list removing alpha numeric
    text_char_sp = [char for char in text_char if not(char.isalnum())]
    
    return text_char_sp 



# ----------------------------------------------------------------------------------------------
def strip_additional_characters(text): 
    """handles the stripping of non alpha-numeric characters for a list of strings"""
    
    special = get_special_chars(text)

    # define characters you want to retain
    char_keep = [' ', '#', '+', '\n', '/']

    char_set = set([c for c in special if c not in char_keep])

    stripped = ' '.join(''.join([c for c in item if c not in char_set]) for item in text.split())

    return stripped