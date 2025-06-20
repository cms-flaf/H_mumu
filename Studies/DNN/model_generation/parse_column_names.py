def parse_column_names(config, column_type="all"):
    """
    Takes a config file specifying which columns to save in the NN sample sets.
    Generally this would be ds_setup/general.yaml.
    Returns a list of all the (unique) column names.
    """
    col_to_save = []
    muon_vars = config["Muon"]
    VBFJet_vars = config["VBFJet"]
    for mu_idx in [1, 2]:
        for mu_var in muon_vars:
            col_to_save.append(mu_var.format(mu_idx))
    for j_idx in [1, 2]:
        for VBFJ_var in VBFJet_vars:
            col_to_save.append(f"j{j_idx}_{VBFJ_var}")
    # SelectedJet_vars = config['vars_to_save']['SelectedJet']
    # for Jvar in SelectedJet_vars:
    #     col_to_save.append(f"SelectedJet_{Jvar}")
    # SoftJet_vars = config["vars_to_save"]["SoftJet"]
    # for SJvar in SoftJet_vars:
    #    col_to_save.append(f"SoftJet_{SJvar}")
    VBFJetPair_vars = config["VBFJetPair"]
    MuJet_vars = config["MuJet"]
    MuPair_vars = config["MuPair"]
    Global_vars = config["Global"]
    Category_vars = config["Categories"]
    Region_vars = config["Regions"]
    Sign_vars = config["Sign"]
    # nJet_vars = config["nJets"]
    Weight_vars = config["Weight"]

    data_columns = VBFJetPair_vars + MuJet_vars + MuPair_vars
    header_columns = Global_vars + Weight_vars
    selection_columns = Sign_vars + Region_vars + Category_vars

    if column_type == "all":
        return header_columns + selection_columns + data_columns
    elif column_type == "data":
        return data_columns
    elif column_type == "header":
        return header_columns
    elif column_type == "selection":
        return selection_columns
