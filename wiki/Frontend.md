# Frontend Documentation

The frontend is a React application that provides a user-friendly interface for interacting with the LEXISIGHT AI.

## 📂 File Structure

*   `frontend/`
    *   `src/`
        *   `components/`: Reusable UI components.
            *   `ChatInterface.js`: The main chat window where users interact with the bot.
            *   `Header.js`: Navigation bar.
            *   `HeroSection.js`: Landing page component.
            *   `Login.js`, `Signup.js`: Authentication components (if enabled).
        *   `services/`: API integration.
            *   `api.js`: Handles HTTP requests to the FastAPI backend.
        *   `styles/`: CSS files for components.
        *   `App.js`: Main application component and routing.
        *   `index.js`: Entry point.

## ⚛️ Technology Stack

*   **React**: UI library.
*   **Axios/Fetch**: For making HTTP requests to the backend.
*   **CSS**: Styling.

## 🔄 Workflow

1.  **User Input**: The user enters a query about a contract dispute in `ChatInterface.js`.
2.  **Part 1 Request**: The frontend sends the query to the `/generate_part1` endpoint.
3.  **Display Questions**: The backend returns clarifying questions, which are displayed to the user.
4.  **User Answers**: The user provides answers to these questions.
5.  **Part 2 Request**: The frontend sends the original query, case summary, and user answers to `/generate_part2`.
6.  **Display Analysis**: The backend returns the structured legal analysis, which is formatted and displayed.

## 🔌 API Integration

The frontend communicates with the backend via the URL provided by ngrok.
**Important**: You must update the base URL in `src/services/api.js` (or equivalent) whenever the backend ngrok tunnel is restarted.
