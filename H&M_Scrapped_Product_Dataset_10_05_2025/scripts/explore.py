import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from utils import grab_col_names, check_traits

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Get current date & format to YYYY-MM-DD
now = datetime.datetime.now()
current_date = now.strftime("%Y_%m_%d")

base_path = Path(__file__).parents[1]
# Set the path to the data directory
data_dir = base_path / "data"
results_dir = base_path / "results"
intermediate_dir = results_dir / "intermediate" / current_date
tables_dir = results_dir / "tables" / current_date
plot_dir = results_dir / "plots" / current_date
# Create directories if they don't exist
intermediate_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# Load the dataset
csv_path = data_dir / "handm.csv"
df_original = pd.read_csv(csv_path, index_col=0)
df = df_original.copy()
df.drop(columns=[
    "brandName", "stockState", "comingSoon", "isOnline",
    "url", "colorName"
], inplace=True, errors='ignore')
df.dropna(inplace=True)
df.set_index("productId", inplace=True)
grab_col_names(df)

# Plot distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid()
plt.savefig(plot_dir / "price_distribution.png")

# We observe that the price distribution is left-skewed, 
# indicating that most products are priced lower,
# with a few high-priced items
# we transform the price to a log scale
df["log_price"] = np.log1p(df["price"])
plt.figure(figsize=(10, 6))
sns.histplot(df["log_price"], bins=50, kde=True)
plt.title("Log Price Distribution")
plt.xlabel("Log Price")
plt.ylabel("Frequency")
plt.grid()
plt.savefig(plot_dir / "log_price_distribution.png")


# Now we look at individual columns
# productNames
plt.figure(figsize=(10, 6))
# Plot the top 20 most common product names
top_product_names = df["productName"].value_counts().nlargest(20)
top_product_names.plot(kind='bar')
plt.title("Top 20 Product Names")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.savefig(plot_dir / "top_20_product_names.png")

# We plot the products with the highest average price
plt.figure(figsize=(10, 6))
# Plot the top 20 product names with the highest average price
top_product_names_avg_price = df.groupby("productName")["price"].mean().nlargest(20)
top_product_names_avg_price.plot(kind='bar')
plt.title("Top 20 Product Names with Highest Average Price")
plt.ylabel("Average Price")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.savefig(plot_dir / "top_20_product_names_avg_price.png")
# Interesting leather products are among the most expensive

# We stratify the product names now based on prior knowledge
product_categories = ["shoes", "slipper", "jacket", "sneaker", "boot", 
                      "loafer", "mule", "pump", "sandal", "heal",
                      "hoodie", "shirt", "jeans", "jogger", "sweater", "pants",
                      "top", "coat", "flats","slingback", "espadrilles",
                      "chinos", "socks", "vest", "shacket", "short", "cardigan",
                      "windbreaker", "bathrobe", "pajamas", "tights", "blazer", "short",
                      "parka", "blouse", "dress", "skirt", "hat", "legging", "beanie", 
                      "bikini", "hair claw", "earrings", "bodysuit", "swimsuit", "jumpsuit",
                      "bracelet", "necklace", "bag", "scarf", "gloves", "belt",
                      "bra", "jumper", "anorak", "briefs", "googles", "cap", "sunglasses", 
                      "lip gloss", "headband", "skort", "leg warmer", "scrunchie",
                      "backpack", "nail polish", "nail clipper", "brush", "mascara",
                      "umbrella", "concealer", "jegging", "ski suit", "warmer", "balaclava",
                      "earmuffs", "slacks", "eyeshadow", "trouser"]

df["productName"] = df["productName"].str.lower()
df["productCategory"] = df["productName"].apply(lambda x: check_traits(x, product_categories))
df["productCategory"] = df["productCategory"].astype("category") 
df["productCategory"] = (df["productCategory"].
                            cat.
                            reorder_categories(
                                list((df.
                                      groupby("productCategory")["price"].
                                      median().
                                      sort_values(ascending=False)).
                                      index)))
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="productCategory", y="price")
plt.ylabel("Price")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.savefig(plot_dir / "price_by_product_category.png")
# We observe that nail clippers are least expensive 
# whereas ski suits are most expensive. This is expected &
# a good quality check
gender_attributes = ["women", "ladies", "men"]
# We can now have a look at male and female product stratification
df["mainCatCode"] = df["mainCatCode"].str.lower()
df["gender"] = df["mainCatCode"].apply(lambda x: check_traits(x, gender_attributes))
gender_map = {"men": "Men", "women": "Women", "ladies": "Women", "Other": "Other"}
# Apply mapping; unmatched values become NaN
df["gender"] = df["gender"].map(gender_map)
df["gender"] = df["gender"].astype("category") 
plt.figure(figsize=(20, 6))
sns.boxplot(data=df[df["gender"].isin(["Men", "Women"])], 
            x="productCategory", y="price", hue="gender")
plt.ylabel("Price")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.savefig(plot_dir / "price_by_product_category_and_gender.png")

df_save = df.copy()
df_save = df_save[["productCategory", "gender", "log_price"]]
df_save = df_save[df_save["productCategory"]!="Other"]
df_save.to_csv(intermediate_dir / "train_test.csv")

sub_df = df[df["productCategory"] == "ski suit"]

# We now have to be creative to decode
# additional properties for each product category
# we start by testing whether there are obvious differences in the 
# pricing based on the colors
# For that we have to convert the hex colors to rgb giving an (N,3) 
# feature vector
def hex_to_rgb(hex_color):
    # Remove the '#' character if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return r, g, b
# Convert the hex color to RGB
df["colors"] = df["colors"].apply(lambda x: "#" + x)
df["colors"] = df["colors"].apply(hex_to_rgb)
# Split the RGB values into separate columns
df[["R", "G", "B"]] = pd.DataFrame(df["colors"].tolist(), index=df.index)
# Make paired scatter plots using sns pairplot
plt.figure(figsize=(10, 6))
sns.pairplot(df, x_vars=["R", "G", "B"], y_vars=["price"], height=5)
plt.suptitle("Price vs RGB Colors", y=1.02)
plt.savefig(plot_dir / "price_vs_rgb_colors.png")
# We observe that there is no clear correlation between the colors and the price 
# at least not on a "global" level, maybe when we look at the individual products
# except for charcoal colors which are more expensive, could be because some 
# products appear only in charcoal colors
df[df["productName"]=="fleece-lined slippers"]
# When we check this we do observe this effect on a product by product basis
# TO-DO: Check whether there is way to actually describe & model this better

# Let's now lock at the materials
# only keep the lines that inform on the components
import pandas as pd
import re

def extract_components(materials: str, product_name: str) -> pd.DataFrame:
    lines = materials.split("\n")
    
    # Filter lines with ":" and no surrounding whitespace
    filtered = list(filter(lambda x: ":" in x and not re.search(r"\s:\s|\s:|:\s", x), lines))
    
    # Take first part before comma, strip spaces
    filtered = [line.split(",")[0].strip() for line in filtered]
    
    # Keep only lines ending in a percentage
    filtered = [line for line in filtered if re.search(r" \d{1,3}%$", line)]

    rows = []
    for line in filtered:
        try:
            component, rest = line.split(":", 1)
            match = re.search(r"(.+?)\s(\d{1,3}%)$", rest.strip())
            if match:
                material = match.group(1).strip()
                percentage = match.group(2)
                rows.append((component.strip(), material, percentage))
        except ValueError:
            continue  # skip malformed lines

    df_components = pd.DataFrame(rows, columns=["component", "material", "percentage"])
    df_components["productName"] = product_name
    return df_components

# Collect all extracted DataFrames
all_components = []

for idx, row in df.iterrows():
    result = extract_components(row["materials"], row["productName"])
    all_components.append(result)

# Concatenate all into a single DataFrame
df_result = pd.concat(all_components, ignore_index=True)
# Add the price column to the DataFrame
df_result = pd.merge(df_result, df[["productName", "price"]], on="productName", how="left")
# Plot whether there are obvious differences in the pricing based on the materials
df_plot = df_result.groupby("material")["price"].mean().reset_index()
df_plot = df_plot.sort_values(by="price", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(df_plot, x="material", y="price")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.savefig(plot_dir / "price_by_material.png")

# Add this information to the original dataset
# I want to add whether a product contains a certain material
materials = df_result["material"].unique()
for i, row in df.iterrows():
    # Check if the product name exists in df_result
    if row["productName"] in df_result["productName"].values:
        # Get the materials for this product
        materials_for_product = df_result[df_result["productName"] == row["productName"]]["material"].tolist()
        # Join them into a single string
        df.at[i, "materials"] = ", ".join(materials_for_product)
    else:
        df.at[i, "materials"] = None
# Now we make the materials unique & expand the feature columsn into "has_":
materials = df["materials"].str.get_dummies(sep=", ")

df_save = df.copy()
df_save = df_save[["productCategory", "gender"]]
df_save = df_save[df_save["productCategory"]!="Other"]
df_save = pd.get_dummies(df_save, drop_first=True)
df_save["log_price"] = df["log_price"]
df_save = pd.concat([df_save, materials], axis=1)
df_save = df_save*1
df_save.to_csv(intermediate_dir / "train_test.csv")

# We can now parse the details for each product
df["descriptions"] = df["details"].apply(lambda x: x.split("\n")[0])
