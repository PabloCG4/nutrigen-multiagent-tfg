import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from src.text_engine import config

# --- Path Configuration ---
OUTPUT_IMG_DIR = config.FIGURES_DIR

# --- Extended Keyword Dictionaries ---
KEYWORDS = {
    'MEAT': [
        'pollo', 'carne', 'ternera', 'cerdo', 'panceta', 'jamón', 'pavo', 'cordero', 
        'bacon', 'salchicha', 'chorizo', 'solomillo', 'costilla', 'albóndiga', 'hamburguesa', 'bistec',
        'chicken', 'beef', 'pork', 'ham', 'lamb', 'steak', 'turkey', 'sausage', 'meat', 
        'ribs', 'pepperoni', 'duck', 'rabbit', 'veal', 'burger', 'meatball'
    ],
    'FISH': [
        'pescado', 'atún', 'salmón', 'merluza', 'bacalao', 'gamba', 'langostino', 'marisco', 
        'lubina', 'pulpo', 'calamar', 'sepia', 'mejillón', 'almeja', 'anchoa', 'sardina', 'dorada',
        'fish', 'tuna', 'salmon', 'cod', 'shrimp', 'prawn', 'crab', 'lobster', 'bass', 'seafood',
        'octopus', 'squid', 'calamari', 'mussel', 'clam', 'scallop', 'anchovy', 'sardine', 'trout'
    ],
    'VEGGIE': [
        'vegetariano', 'vegano', 'verdura', 'tofu', 'seitan', 'tempeh', 'lenteja', 'garbanzo', 
        'judía', 'ensalada', 'huerta', 'soja', 'vegetal', 'espinaca', 'brócoli', 'coliflor',
        'vegetarian', 'vegan', 'vegetable', 'tofu', 'seitan', 'tempeh', 'lentil', 'chickpea', 
        'bean', 'salad', 'plant-based', 'soy', 'spinach', 'broccoli', 'cauliflower', 'kale'
    ],
    'DESSERT': [
        'chocolate', 'azúcar', 'sugar', 'tarta', 'pastel', 'cake', 'cookie', 'galleta', 
        'postre', 'dessert', 'dulce', 'sweet', 'vanilla', 'vainilla', 'muffin', 'brownie',
        'bizcocho', 'helado', 'ice cream', 'pudding'
    ],
    'APPLIANCES': {
        'Oven': ['horno', 'hornear', 'asar', 'oven', 'bake', 'roast', 'preheat'],
        'Microwave': ['microondas', 'microwave'],
        'Pan/Skillet': ['sartén', 'sarten', 'freír', 'saltear', 'pan', 'skillet', 'fry', 'saute', 'stir-fry'],
        'Air Fryer': ['freidora de aire', 'air fryer', 'airfryer']
    },
    'ORIGIN': {
        'Spanish': ['española', 'spanish', 'paella', 'gazpacho', 'tortilla', 'tapas'],
        'Italian': ['italiana', 'italian', 'pasta', 'pizza', 'risotto', 'lasagna', 'spaghetti'],
        'Mexican': ['mexicana', 'mexican', 'taco', 'burrito', 'enchilada', 'fajita', 'guacamole', 'quesadilla'],
        'Asian': ['asiática', 'asian', 'china', 'chinese', 'japonesa', 'japanese', 'sushi', 'wok', 'ramen', 'thai', 'curry'],
        'Indian': ['india', 'indian', 'tikka', 'masala', 'naan']
    }
}

# --- Helper Functions ---

def calculate_list_length(string_representation):
    """Parses a stringified list and returns its length."""
    if pd.isna(string_representation):
        return 0
    try:
        actual_list = ast.literal_eval(str(string_representation))
        if isinstance(actual_list, list):
            return len(actual_list)
        return 0
    except (ValueError, SyntaxError):
        return 0

def classify_dietary_type(row):
    """
    Determines diet type using a Priority Waterfall logic.
    """
    text_content = (str(row['title']) + " " + str(row['final_ingredients'])).lower()
    
    # 1. Check for Desserts
    for word in KEYWORDS['DESSERT']:
        if word in text_content:
            return 'Dessert'

    # 2. Check for Meat
    for word in KEYWORDS['MEAT']:
        if word in text_content:
            return 'Meat'
    
    # 3. Check for Fish/Seafood
    for word in KEYWORDS['FISH']:
        if word in text_content:
            return 'Fish'
            
    # 4. Check for EXPLICIT Vegetarian keywords
    for word in KEYWORDS['VEGGIE']:
        if word in text_content:
            return 'Vegetarian'

    # 5. Fallback
    return 'Other'

def classify_origin_region(title):
    """Detects geographical origin based on title keywords."""
    title_lower = str(title).lower()
    for region, tags in KEYWORDS['ORIGIN'].items():
        for tag in tags:
            if tag in title_lower:
                return region
    return 'General'

def detect_appliance_usage(directions_text, tags):
    """Returns 1 if any tag is found in the text, else 0."""
    text_str = str(directions_text).lower()
    for tag in tags:
        if tag in text_str:
            return 1
    return 0

# --- Main Analysis Logic ---

def analyze_recipes():
    if not os.path.exists(config.MASTER_RECIPE_PATH):
        print(f"Error: Master dataset not found at {config.MASTER_RECIPE_PATH}")
        return

    print("Loading master dataset (this might take a moment)...")
    df = pd.read_csv(config.MASTER_RECIPE_PATH)
    
    # ---------------------------------------------------------
    # 1. DATA PREPROCESSING
    # ---------------------------------------------------------
    print("Computing statistics...")
    
    # Calculate counts
    df['step_count'] = df['structured_directions'].apply(calculate_list_length)
    df['ing_count'] = df['final_ingredients'].apply(calculate_list_length)
    
    # Classify Diet
    print("Classifying diets...")
    df['diet_type'] = df.apply(classify_dietary_type, axis=1)

    # Classify Origin (New for the extra chart)
    print("Classifying origins...")
    df['origin_region'] = df['title'].apply(classify_origin_region)

    # Detect Appliances
    print("Detecting appliances...")
    for appliance_name, tags_list in KEYWORDS['APPLIANCES'].items():
        # Pass tags_list as an argument to the apply function
        df[appliance_name] = df['structured_directions'].apply(detect_appliance_usage, args=(tags_list,))

    # ---------------------------------------------------------
    # 2. VISUALIZATION GENERATION
    # ---------------------------------------------------------
    # Set global style
    sns.set_theme(style="whitegrid", context="talk")
    
    # --- CHART 1: INGREDIENT COUNT DISTRIBUTION ---
    print("Generating Ingredient Distribution Histogram...")
    plt.figure(figsize=(12, 7))
    
    # Filter reasonable counts for better visualization
    filtered_data = df[df['ing_count'] <= 30]
    
    sns.histplot(data=filtered_data, x='ing_count', kde=True, bins=20, color="#3498db", element="step")
    plt.title("Distribution of Ingredient Counts per Recipe")
    plt.xlabel("Number of Ingredients")
    plt.ylabel("Frequency (Number of Recipes)")
    plt.axvline(filtered_data['ing_count'].mean(), color='red', linestyle='--', label=f"Mean: {filtered_data['ing_count'].mean():.1f}")
    plt.legend()
    
    plot1_path = os.path.join(OUTPUT_IMG_DIR, "ingredient_distribution_histogram.png")
    plt.savefig(plot1_path)
    print(f" > Saved: {plot1_path}")
    plt.close()

    # --- CHART 2: DIET DISTRIBUTION ---
    print("Generating Diet Distribution Chart...")
    plt.figure(figsize=(12, 6))
    
    order_list = ['Meat', 'Fish', 'Vegetarian', 'Dessert', 'Other']
    diet_counts = df['diet_type'].value_counts().reindex(order_list, fill_value=0)
    
    ax = sns.barplot(x=diet_counts.index, y=diet_counts.values, palette="Spectral")
    plt.title(f"Recipe Distribution by Diet Category (Total: {len(df):,})")
    plt.ylabel("Count")
    plt.bar_label(ax.containers[0], fmt='%.0f')
    
    plot2_path = os.path.join(OUTPUT_IMG_DIR, "diet_distribution_bar.png")
    plt.savefig(plot2_path)
    print(f" > Saved: {plot2_path}")
    plt.close()

    # --- CHART 3: APPLIANCE USAGE BY DIET ---
    print("Generating Appliance Usage Chart...")
    appliance_cols = list(KEYWORDS['APPLIANCES'].keys())
    
    # Filter to main categories
    df_clean = df[df['diet_type'].isin(['Meat', 'Fish', 'Vegetarian', 'Dessert'])]
    
    appliance_data = df_clean.groupby('diet_type')[appliance_cols].sum().reset_index()
    appliance_melted = appliance_data.melt(id_vars='diet_type', var_name='Appliance', value_name='Frequency')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=appliance_melted, x='Appliance', y='Frequency', hue='diet_type', palette="Set2")
    plt.title("Appliance Usage by Diet Type")
    plt.xticks(rotation=45)
    plt.legend(title='Category')
    plt.tight_layout()
    
    plot3_path = os.path.join(OUTPUT_IMG_DIR, "appliance_usage_by_diet.png")
    plt.savefig(plot3_path)
    print(f" > Saved: {plot3_path}")
    plt.close()

    # --- CHART 4: Complexity by Origin ---
    # Boxplot showing the distribution of steps for different cuisines.
    # This answers: "Are Asian recipes more complex than Spanish ones?"
    print("Generating Complexity vs Origin Chart...")
    
    plt.figure(figsize=(14, 8))
    
    # Filter out 'General' to focus on specific cuisines
    df_origin = df[df['origin_region'] != 'General']
    
    # Remove extreme outliers for readability (recipes with > 40 steps)
    df_origin_clean = df_origin[df_origin['step_count'] < 40]

    # Sort origins by median step count for cleaner visualization
    if not df_origin_clean.empty:
        order_by_median = df_origin_clean.groupby("origin_region")["step_count"].median().sort_values().index
        
        sns.boxplot(x='origin_region', y='step_count', data=df_origin_clean, order=order_by_median, palette="magma")
        plt.title("Recipe Complexity (Number of Steps) by Cultural Origin")
        plt.ylabel("Number of Steps per Recipe")
        plt.xlabel("Cuisine Origin")
        
        plot4_path = os.path.join(OUTPUT_IMG_DIR, "complexity_by_origin_boxplot.png")
        plt.savefig(plot4_path)
        print(f" > Saved: {plot4_path}")
    else:
        print(" > Skipped Chart 4: Not enough origin data detected.")
        
    plt.close()

    print("\n--- Analysis Completed ---")

if __name__ == "__main__":
    analyze_recipes()