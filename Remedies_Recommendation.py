import json

with open("health_remedies.json", "r", encoding="utf-8") as file:
    remedies_data = json.load(file)
def get_remedy_from_json(disease_name):
    disease_name_lower = disease_name.lower().strip()
    for issue in remedies_data["health_issues"]:
        if issue["name"].lower() == disease_name_lower:
            return issue["remedy"]  
    for issue in remedies_data["health_issues"]:
        if disease_name_lower in issue["name"].lower():  
            return issue["remedy"]
    return None 
def get_remedy(disease_name):
    remedy_steps = get_remedy_from_json(disease_name)
    if remedy_steps:
        print(f"\nRemedy for {disease_name}:\n")
        for i, step in enumerate(remedy_steps, 1):
            print(f"{i}. {step}\n")  
    else:
        print(f"\n⚠ No remedy found for '{disease_name}' in the database.")
disease_input = input("Enter a disease name: ").strip()
get_remedy(disease_input)
