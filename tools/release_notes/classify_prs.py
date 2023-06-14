# In[1]:


import pandas as pd


# In[2]:


# from https://github.com/pytorch/audio/blob/main/.github/process_commit.py
primary_labels_mapping = {
    "BC-breaking": "Backward-incompatible changes",
    "deprecation": "Deprecations",
    "bug fix": "Bug Fixes",
    "new feature": "New Features",
    "improvement": "Improvements",
    "prototype": "Prototypes",
    "other": "Other",
    "None": "Missing",
}

secondary_labels_mapping = {
    "module: io": "I/O",
    "module: ops": "Ops",
    "module: models": "Models",
    "module: pipelines": "Pipelines",
    "module: datasets": "Datasets",
    "module: docs": "Documentation",
    "module: tests": "Tests",
    "tutorial": "Tutorials",
    "recipe": "Recipes",
    "example": "Examples",
    "build": "Build",
    "style": "Style",
    "perf": "Performance",
    "other": "Other",
    "None": "Missing",
}


# In[3]:


df = pd.read_json("data.json").T
df.tail()


# In[4]:


def get_labels(col_name, labels):
    df[col_name] = [[] for _ in range(len(df))]
    for _, row in df.iterrows():
        row[col_name] = "None"
        for label in labels:
            if label in row["labels"]:
                row[col_name] = label
                break


# In[5]:


get_labels("primary_label", primary_labels_mapping.keys())
get_labels("secondary_label", secondary_labels_mapping.keys())
df.tail(5)


# In[6]:


for primary_label in primary_labels_mapping.keys():
    primary_df = df[df["primary_label"] == primary_label]
    if primary_df.empty:
        continue
    print(f"## {primary_labels_mapping[primary_label]}")
    for secondary_label in secondary_labels_mapping.keys():
        secondary_df = primary_df[primary_df["secondary_label"] == secondary_label]
        if secondary_df.empty:
            continue
        print(f"### {secondary_labels_mapping[secondary_label]}")
        for _, row in secondary_df.iterrows():
            print(f"- {row['title']}")
        print()
    print()
