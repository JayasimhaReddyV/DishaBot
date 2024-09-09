# DishaBOT - A chatbot that guides your steps and shapes your future
# implementing using the Natural Language ToolKit (nltk)

# Import necessary libraries required for the application
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data files to meet up neccessary requirements if needed
# nltk.download('punkt')
#nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize stop words and lemmatizer as toola for Natural Language Processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a dictionary of questions and their corresponding responses
qa_pairs = {
    "Hi ,hello, hello?, hey": "Hey Buddy! I am DISHABOT. I'm here to offer you guidance related to your career!! How can I assist you today?",
    "What technical skills should I focus on learning to enhance my career prospects?": "It’s crucial to focus on programming languages such as Python, Java, or C++. Also, consider learning data structures, algorithms, and skills related to data analysis, cloud computing, and machine learning.",
    "How can I find an internship in the tech industry?": "Start by exploring internship opportunities on job portals like MACHIT, LinkedIn, Glassdoor, and Indeed. Network through professional platforms, attend career fairs, and directly contact companies you’re interested in.",
    "What are the most in-demand programming languages currently?": "Some of the most in-demand programming languages include Python, JavaScript, Java, C#, and SQL. It’s beneficial to have proficiency in at least one of these languages.",
    "How do I create an impressive tech resume?": "Highlight your technical skills, include relevant projects and internships, emphasize your experience with specific tools and languages, and use quantifiable achievements to demonstrate your impact.",
    "What are some essential soft skills for a successful tech career?": "Key soft skills include communication, problem-solving, teamwork, adaptability, and time management. These skills help you work effectively within a team and adapt to changing technologies.",
    "Can you recommend any online courses for learning new technologies?": "Platforms like Coursera, Udemy, edX, and Khan Academy offer a variety of courses in programming, data science, AI, and other technologies. Many of these courses are flexible and self-paced.",
    "What are some good projects to include in my portfolio?": "Consider projects that showcase your skills in web development, mobile app creation, data analysis, or any open-source contributions. Projects that solve real-world problems are particularly impressive.",
    "Should I focus on learning a specific technology or multiple technologies?": "Start by mastering one technology deeply, but also have a basic understanding of others. This makes you versatile and more adaptable to different roles and projects.",
    "What are some good networking strategies in the tech industry?": "Attend industry conferences, join professional groups on LinkedIn, participate in hackathons, and engage in online forums like GitHub and Stack Overflow to connect with peers and professionals.",
    "How can I stay updated with the latest tech trends?": "Follow tech blogs, subscribe to industry newsletters, join relevant LinkedIn groups, and participate in webinars and online courses to stay informed about new developments and innovations.",
    "What are the best ways to develop problem-solving skills in tech?": "Work on coding challenges, engage in competitive programming, build projects that require troubleshooting, and regularly practice algorithms and data structures.",
    "Should I pursue certifications in my tech field?": "Certifications can validate your skills and knowledge. Consider pursuing certifications in areas like cloud computing (AWS, Azure), cybersecurity, or specific programming languages.",
    "How important is it to contribute to open-source projects?": "Contributing to open-source projects is a great way to gain practical experience, improve your coding skills, and demonstrate your ability to collaborate on real-world projects.",
    "What are some good ways to practice coding regularly?": "Use coding platforms like LeetCode, CodeSignal, or Codewars to practice regularly. You can also contribute to open-source projects or work on your own personal projects.",
    "How can I effectively balance my academic studies and tech skill development?": "Create a schedule that allocates time for both academics and skill development. Prioritize tasks, set specific goals, and use time management tools to stay organized.",
    "What are the most promising fields in technology right now?": "Promising fields include artificial intelligence, machine learning, data science, cybersecurity, cloud computing, and blockchain technology.",
    "What should I consider when choosing a specialization in tech?": "Consider your interests, market demand, career growth opportunities, and the skills required. Research various specializations and evaluate their long-term potential.",
    "What are some effective ways to improve my coding efficiency?": "Practice regularly, write clean and readable code, learn to use development tools and version control systems, and review and refactor your code for improvements.",
    "How can I build a strong professional online presence?": "Create a LinkedIn profile, contribute to tech forums, write blog posts on technical topics, and share your projects and achievements on platforms like GitHub.",
    "What should I include in a tech project presentation?": "Highlight the problem you solved, your approach, the technologies used, and the results achieved. Include demonstrations and code snippets, and be prepared to answer questions.",
    "How can I gain hands-on experience if I’m a beginner?": "Start with small projects, contribute to open-source, participate in coding competitions, take part in internships, and collaborate on projects with peers.",
    "What resources can help me prepare for a tech career transition?": "Utilize career coaching services, attend workshops and webinars, seek advice from industry professionals, and leverage online learning platforms to acquire new skills.",
    "Would you like to ask me any questions?": "Do you have any experience with internships or projects in tech?",
    "Yes, Yeah, I have": "That's great to hear!",
    "No, Nah, Not yet": "That's Fine, we will work into it together!",
    "Thank you, Thanks a lot, You were really helpful": "It was my pleasure helping you... Anything else?",
    "Bye, Good bye, OK I will take a leave now": "Bye, Please type Exit in order to exit the chat!",
    "Would you like to ask me any questions?": "Yes definitely, where do you want to see yourself in next 5 years?",
    "I would like to see myself as software engineer, data analyst, data scientist, entrepreneur in the next 5 years": "That's cool, wish you reach that position!!"    
}

# Preprocess text by tokenizing, removing stop words, and lemmatizing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Preprocess the questions
preprocessed_questions = [preprocess_text(question) for question in qa_pairs.keys()]

# Create a TF-IDF vectorizer and fit it on the preprocessed questions
vectorizer = TfidfVectorizer()
vectorizer.fit(preprocessed_questions)

# Define a similarity threshold
SIMILARITY_THRESHOLD = 0.3

# Function to find the best matching response
def get_response(user_input):
    preprocessed_input = preprocess_text(user_input)
    user_input_vector = vectorizer.transform([preprocessed_input])
    question_vectors = vectorizer.transform(preprocessed_questions)
    
    similarity_scores = cosine_similarity(user_input_vector, question_vectors)
    best_match_index = np.argmax(similarity_scores)
    best_match_score = similarity_scores[0][best_match_index]
    
    if best_match_score < SIMILARITY_THRESHOLD:
        return "Sorry, I couldn't understand your question. Could you please rephrase or ask a different question?"
    
    best_match_question = list(qa_pairs.keys())[best_match_index]
    return qa_pairs[best_match_question]

# Example chatbot conversation
print("DISHABOT: Hello! I'm DISHABOT - A career guidance chatbot.")
print("DISHABOT: Feel free to ask me your doubts you have regarding your Software or Technical career, I'm here to help you!!")
print("DISHABOT: Please respond (Bye/ Exit/ Leave) in order to exit the ongoing chat")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'leave', 'bye']:
        print("DISHABOT: Thank you for reaching out, If you have any other questions, please do reach out!! I'm on all ears ;)")
        break
    response = get_response(user_input)
    print("DISHABOT:", response)
