import os
from supabase import create_client
from dotenv import load_dotenv
from rag_search import run_retrieval_qa  # Import the RAG search function

load_dotenv()

# Initialize Supabase client once here
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

CATEGORY_SYNONYMS = {
    "park": "nature",
    "walk": "nature",
    "hiking": "nature",
    "museum": "culture",
    "art": "culture",
    "food": "restaurant",
    "eat": "restaurant",
    "dining": "restaurant",
    "stay": "hotel",
    "accommodation": "hotel",
}

def get_all_cities():
    results = supabase.table("travel_data").select("metadata").execute()
    data = results.data or []

    cities = set()
    for item in data:
        metadata = item.get("metadata", {})
        city = metadata.get("location")
        if city:
            cities.add(city.lower())
    return sorted(cities)

def get_activity_types_for_city(city):
    results = supabase.table("travel_data") \
        .select("metadata") \
        .filter("metadata->>location", "ilike", f"%{city}%") \
        .execute()
    data = results.data or []

    types = set()
    categories = set()
    for item in data:
        metadata = item.get("metadata", {})
        typ = metadata.get("type")
        cat = metadata.get("category")
        if typ:
            types.add(typ.lower())
        if cat:
            categories.add(cat.lower())
    return types, categories

def get_suggestions(city, activity_type, budget):
    query = supabase.table("travel_data").select("*").filter("metadata->>location", "ilike", f"%{city}%")

    activity_type_lower = activity_type.lower()

    if activity_type_lower in ["hotel", "stay", "accommodation"]:
        query = query.filter("metadata->>type", "eq", "hotel")
    elif activity_type_lower in ["restaurant", "food", "cuisine"]:
        query = query.filter("metadata->>type", "eq", "restaurant")
    else:
        query = query.filter("metadata->>type", "eq", "attraction") \
                     .filter("metadata->>category", "ilike", f"%{activity_type_lower}%")

    results = query.execute()
    data = results.data or []

    if not data:
        return []

    def price_ok(record):
        try:
            price = float(record["metadata"].get("price", 0))
        except Exception:
            return True

        if budget == "low":
            return price <= 30
        elif budget == "medium":
            return 30 < price <= 200
        elif budget == "high":
            return price > 200
        else:
            return True

    filtered = [r for r in data if price_ok(r)]
    return filtered

def run_itinerary_planner():
    cities = get_all_cities()
    while True:
        city = input("Which city are you going to? ").strip().lower()
        if city not in cities:
            print(f"Sorry, we currently don't have data for '{city.title()}'. Available cities are:")
            for c in cities:
                print(f"- {c.title()}")
        else:
            break

    available_types, available_categories = get_activity_types_for_city(city)
    available_activities = set()
    if "hotel" in available_types:
        available_activities.update(["hotel", "stay", "accommodation"])
    if "restaurant" in available_types:
        available_activities.update(["restaurant", "food", "cuisine"])
    available_activities.update(available_categories)

    while True:
        activity = input("What type of activity are you interested in? (e.g., hotel, restaurant, nature, park, history) ").strip().lower()
        activity = CATEGORY_SYNONYMS.get(activity, activity)

        if activity not in available_activities:
            print(f"Sorry, no '{activity}' activities found in {city.title()}. Please choose from:")
            for a in sorted(available_activities):
                print(f"- {a}")
            continue

        while True:
            budget = input("What's your budget? (low, medium, high) ").strip().lower()
            suggestions = get_suggestions(city, activity, budget)

            if not suggestions:
                print(f"Sorry, no results found for {activity} in {city.title()} within your budget.")
                retry = input("Would you like to try a different activity or budget? (yes/no) ").strip().lower()
                if retry in ["yes", "y"]:
                    change = input("Type 'activity' to change activity or 'budget' to change budget: ").strip().lower()
                    if change == "activity":
                        break  # back to activity input
                    elif change == "budget":
                        continue  # retry budget input for same activity
                    else:
                        print("Invalid choice, exiting planner.")
                        return
                else:
                    print("Okay, exiting planner.")
                    return
            else:
                print(f"\nTop suggestions for {activity} in {city.title()} (budget: {budget}):\n")
                for i, s in enumerate(suggestions, 1):
                    content = s.get("content") or s.get("metadata", {}).get("description") or "No description"
                    price = s["metadata"].get("price", "N/A")
                    currency = s["metadata"].get("currency", "")
                    print(f"{i}. {content} — Price: {price} {currency}")
                return  # done with planner

def run_retrieval_qa(supabase):
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    vectorstore = SupabaseVectorStore.from_texts(
        texts=[d["content"] for d in supabase.table("travel_vectors").select("content").execute().data],
        embedding=embeddings,
        client=supabase,
        table_name="travel_vectors"
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    print("\nWelcome to the Travel Assistant Q&A! Type 'exit' to return to main menu.")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit']:
            print("Returning to main menu...\n")
            break
        answer = qa.invoke(query)
        if isinstance(answer, dict) and "result" in answer:
            print("\nAnswer:", answer["result"])
        else:
            print("\nAnswer:", answer)

        # Prompt to ask another question or exit
        while True:
            cont = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
            if cont in ['yes', 'y']:
                break  # ask another question
            elif cont in ['no', 'n']:
                print("Returning to main menu...\n")
                return  # exit ask mode back to main menu
            else:
                print("Please enter yes or no.")

def main():
    print("Welcome! You can plan your trip or ask travel questions.")

    while True:
        choice = input("Type 'plan' to get itinerary suggestions, 'ask' to ask a question, or 'exit' to quit: ").strip().lower()

        if choice == "plan":
            run_itinerary_planner()
        elif choice == "ask":
            run_retrieval_qa(supabase)
        elif choice == "exit":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 'plan', 'ask', or 'exit'.")

if __name__ == "__main__":
    main()
