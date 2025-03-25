# Process Conditions
import re

def modify_condition(morphoframe, condition, before, after):
    if condition not in morphoframe.keys():
        print("%s not in morphoframe..."%condition)
    print("Replacing all instances of `%s` in the `%s` morphoframe column with %s"%(before, condition, after))
    morphoframe.loc[morphoframe[condition].isin(before), condition] = after
    return morphoframe

def _drop_condition(morphoframe, condition, values):
    for value in values:
        print("Filtering out %s from %s..."%(value, condition))
        if value not in morphoframe[condition].unique():
            print("%s not in the available condition..."%value)
        else:
            morphoframe = morphoframe.loc[~morphoframe[condition].str.contains(re.escape(value))].reset_index(drop=True)
    return morphoframe

def _keep_condition(morphoframe, condition, values):
    value = "|".join(values)
    print("Keeping %s in %s..."%(value, condition))
    morphoframe = morphoframe.loc[morphoframe[condition].str.contains(value)].reset_index(drop=True)
    return morphoframe

def restrict_conditions(morphoframe, condition, values, action):
    if condition not in morphoframe.keys():
        print("%s not in morphoframe..."%condition)
    else:
        if action == "drop":
            morphoframe = _drop_condition(morphoframe, condition, values)
        elif action == "keep":
            morphoframe = _keep_condition(morphoframe, condition, values)
        else:
            print("Warning for ", condition, values)
            print("Third column must be either 'drop' or 'keep'...")
            print("Nothing will be done...")
    return morphoframe
