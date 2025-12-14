import React, { useState, useRef, useEffect } from 'react';
import '../styles/ChatInterface.css';
import { legalAssistantApi } from '../services/api';

// Helper function to render text with preserved line breaks
const TextWithLineBreaks = ({ text }) => {
  if (!text) return null;
  
  // Split text by double line breaks
  const paragraphs = text.split(/\n\n+/);
  
  return (
    <>
      {paragraphs.map((paragraph, i) => (
        <p key={i} className="analysis-paragraph">{paragraph}</p>
      ))}
    </>
  );
};

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPhase, setCurrentPhase] = useState('initial'); // initial, questions, answers, analysis
  const [questions, setQuestions] = useState([]);
  const [caseSummary, setCaseSummary] = useState('');
  const [answers, setAnswers] = useState([]);
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Handle sending the initial query
  const handleSendQuery = async () => {
    if (!input.trim()) return;
    
    // Add user message to chat
    const userMessage = { type: 'user', content: input };
    setMessages([...messages, userMessage]);
    
    // Store the query for later use
    const query = input;
    setInput('');
    setLoading(true);

    try {
      // Call API to get questions (Part 1)
      const response = await legalAssistantApi.generateQuestions(query);
      
      // Store case summary and questions
      setCaseSummary(response.case_summary);
      setQuestions(response.questions);
      setAnswers(new Array(response.questions.length).fill(''));
      
      // Add case summary to chat
      setMessages(msgs => [...msgs, {
        type: 'assistant',
        content: `Based on your query, here's a brief case summary:\n\n${response.case_summary}`
      }]);
      
      // Add questions to chat
      setMessages(msgs => [...msgs, {
        type: 'assistant',
        content: 'To provide better legal assistance, I need some additional information:',
        questions: response.questions
      }]);
      
      // Update phase
      setCurrentPhase('questions');
    } catch (error) {
      setMessages(msgs => [...msgs, {
        type: 'system',
        content: 'Sorry, there was an error processing your request. Please try again.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Handle answering questions
  const handleAnswerQuestion = async (questionIdx, answer) => {
    // Update answers array
    const newAnswers = [...answers];
    newAnswers[questionIdx] = answer;
    setAnswers(newAnswers);
    
    // Add user answer to chat
    setMessages(msgs => [...msgs, {
      type: 'user',
      content: answer,
      isAnswer: true,
      questionIdx
    }]);
    
    // Check if all questions are answered
    const allAnswered = newAnswers.filter(a => a && a.trim()).length === questions.length;
    
    if (allAnswered) {
      setCurrentPhase('answers');
      setLoading(true);
      
      try {
        // Get stored query
        const query = messages.find(m => m.type === 'user' && !m.isAnswer)?.content || '';
        
        // Call API for legal analysis (Part 2)
        const analysisResponse = await legalAssistantApi.generateAnalysis(query, caseSummary, newAnswers);
        
        // Add analysis sections to chat
        setMessages(msgs => [...msgs, {
          type: 'assistant',
          content: 'Based on the information provided, here is my legal assessment:',
          analysis: analysisResponse
        }]);
        
        setCurrentPhase('analysis');
      } catch (error) {
        setMessages(msgs => [...msgs, {
          type: 'system',
          content: 'Sorry, there was an error generating the legal analysis. Please try again.'
        }]);
      } finally {
        setLoading(false);
      }
    }
  };

  // Handle submitting an answer in the input field
  const handleSubmitAnswer = () => {
    if (!input.trim()) return;
    
    // Find first unanswered question
    const questionIdx = answers.findIndex(a => !a || !a.trim());
    if (questionIdx >= 0) {
      handleAnswerQuestion(questionIdx, input);
      setInput('');
    }
  };

  // Handle input submission based on current phase
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (currentPhase === 'initial' || currentPhase === 'analysis') {
      handleSendQuery();
    } else if (currentPhase === 'questions') {
      handleSubmitAnswer();
    }
  };

  // Handle clicking a suggestion
  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
  };

  // Render message content based on type
  const renderMessageContent = (message) => {
    if (message.type === 'assistant' && message.questions) {
      return (
        <div>
          <p>{message.content}</p>
          <ol className="question-list">
            {message.questions.map((q, idx) => (
              <li key={idx} className={answers[idx] ? 'answered' : ''}>
                {q}
                {answers[idx] && <div className="answer-preview">Your answer: {answers[idx]}</div>}
              </li>
            ))}
          </ol>
        </div>
      );
    } else if (message.type === 'assistant' && message.analysis) {
      return (
        <div className="analysis-container">
          <p>{message.content}</p>
          <div className="analysis-section">
            <h4>Relevant Legal Sections</h4>
            <TextWithLineBreaks text={message.analysis.relevant_legal_sections} />
          </div>
          <div className="analysis-section">
            <h4>Suggested Legal Procedures</h4>
            <TextWithLineBreaks text={message.analysis.suggested_legal_procedures} />
          </div>
          <div className="analysis-section">
            <h4>Strategic Advice</h4>
            <TextWithLineBreaks text={message.analysis.strategic_advice} />
          </div>
          <div className="analysis-section">
            <h4>Estimated Outcome</h4>
            <TextWithLineBreaks text={message.analysis.estimated_outcome} />
          </div>
        </div>
      );
    } else {
      return <p>{message.content}</p>;
    }
  };

  // Show initial suggestions if no messages yet
  const showInitialView = messages.length === 0;

  return (
    <div className="chat-container">
      {!showInitialView && (
        <div className="chat-messages">
          {messages.map((message, idx) => (
            <div key={idx} className={`message ${message.type}-message`}>
              {renderMessageContent(message)}
            </div>
          ))}
          {loading && <div className="message system-message loading">Processing your request...</div>}
          <div ref={messagesEndRef} />
        </div>
      )}
      
      {showInitialView && (
        <div className="chat-box">
          <h2>What Can I Help You With?</h2>
          <div className="suggestions">
            <p onClick={() => handleSuggestionClick("What are the steps I can take towards filing for bankruptcy?")}>
              What are the steps I can take towards filing for bankruptcy?
            </p>
            <p onClick={() => handleSuggestionClick("Someone's pet dog bit me, what can I do?")}>
              Someone's pet dog bit me, what can I do?
            </p>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="chat-input-container">
        <input
          type="text"
          placeholder={
            currentPhase === 'questions' 
              ? `Answer question ${answers.filter(a => a && a.trim()).length + 1} of ${questions.length}...`
              : "Message AILaw..."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="chat-input"
          disabled={loading}
        />
        <button type="submit" className="send-button" disabled={loading}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="feather feather-send"
          >
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
