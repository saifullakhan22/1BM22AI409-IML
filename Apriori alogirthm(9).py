#9


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample dataset (list of transactions)
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Butter', 'Cheese'],
    ['Bread', 'Eggs', 'Cheese'],
    ['Bread', 'Milk', 'Butter']
]

# Convert the dataset into a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display association rules
print("\nAssociation Rules:")
print(rules)
