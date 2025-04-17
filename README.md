🚨 Crime Hotspot Prediction System
📝 Project Description

This project aims to build a Crime Hotspot Detection and Severity Prediction System that analyzes past crime data to identify dangerous areas and predict crime severity levels.
It uses Machine Learning and Clustering Algorithms to detect crime-prone zones ("hotspots") and predict the seriousness of crimes, helping law enforcement and communities improve public safety.

By visualizing crime data on an interactive map and predicting the likelihood of crime incidents, the system serves as a smart early-warning tool for safer city planning and real-time decision-making.
⚙️ Key Features

    Predicts crime severity based on location, time, and crime type

    Detects crime hotspots using clustering algorithms (like K-Means)

    Visualizes hotspots on an interactive heatmap (using Folium)

    Accepts real-time input for new crime incidents

    Provides a simple and user-friendly Flask web interface

📊 Dataset

    Crime incident reports with fields such as:

        Location (latitude, longitude)

        Time of occurrence

        Crime Type (e.g., robbery, assault, theft)

        Severity level (minor, moderate, severe)

    Historical crime datasets were used (example: open city police crime data).

🛠️ Technologies Used

    Python

    Flask (for web application)

    scikit-learn (for machine learning models)

    Pandas (for data preprocessing)

    Folium (for heatmap visualization)

    SQLite (for simple database storage)

    HTML/CSS (for frontend design)

🧠 Machine Learning Models

    KMeans Clustering:
    For identifying hotspots based on crime density.

    Support Vector Machine (SVM):
    For predicting the severity of crimes based on input features like time, type, and location.

🚀 How the System Works

    Load and preprocess past crime data.

    Use KMeans clustering to find areas with high crime concentrations (hotspots).

    Train an SVM model to predict the severity of new or hypothetical crimes.

    Visualize hotspots dynamically on a heatmap.

    Allow users to input new crime events and instantly get:

        Predicted severity

        Updated heatmap

🎯 Goals of the Project

    Help citizens identify unsafe areas.

    Assist law enforcement in resource allocation.

    Raise public awareness and encourage smarter urban planning.

    Provide early warning for possible crime surges in certain areas.

📈 Results

    Successfully detected real-world crime hotspots using clustering.

    Achieved high accuracy in predicting crime severity.

    Interactive, real-time crime visualization map built and deployed.

🤝 Contribution

Contributions are welcome!
If you have ideas for improvements (e.g., adding real-time crime feeds, better ML models, or advanced map features), feel free to fork the project and send a pull request.
📄 License

This project is licensed under the MIT License — see the LICENSE file for more details.
