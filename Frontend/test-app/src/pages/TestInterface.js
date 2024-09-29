import React, { useState, useEffect } from 'react';
import { useParams, useNavigate} from 'react-router-dom';
import './TestInterface.css'; // Import CSS file

const TestInterface = () => {
  const navigate = useNavigate();
  const { testId } = useParams(); // Get test ID from URL params
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [responses, setResponses] = useState({});
  const [attemptId, setAttemptId] = useState(null); // Store attempt ID
  const [userId, setUserId] = useState(null); // Store userId
  const [testName, setTestName] = useState(''); // Store test name
  const [timer, setTimer] = useState(0); // Store timer (in seconds)
  

  useEffect(() => {
    // Fetch userId from localStorage when component mounts
    const storedUserId = localStorage.getItem('user_id');
    setUserId(storedUserId);

    // Fetch test questions and options from the backend
    const fetchTestData = async () => {
      const response = await fetch('http://localhost:8000/api/start-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_id: testId, user_id: storedUserId }), // Send user ID and test ID
      });
      const data = await response.json();
      if (data.success) {
        setQuestions(data.questions);
        setAttemptId(data.testAttempt.id); // Store attempt ID from response
        setTestName(data.testName); // Store test name
        setTimer(data.testDuration * 60); // Set timer in seconds (duration in minutes * 60)
      }
    };
    fetchTestData();
  }, [testId]);

  // Countdown timer logic
  useEffect(() => {
    if (timer > 0) {
      const countdown = setInterval(() => {
        setTimer(prevTimer => prevTimer - 1);
      }, 1000);
      return () => clearInterval(countdown);
    } else if (timer === 1) {
      handleSubmit(); // Automatically submit when time runs out
    }
  }, [timer]);

  // Convert timer seconds into MM:SS format
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAnswerChange = (questionId, optionId) => {
    setResponses(prevResponses => ({
      ...prevResponses,
      [questionId]: optionId,
    }));
  };

  const handleSubmit = async () => {
    try {
      // Submit responses with attempt ID, test ID, user ID, and answers
      const response = await fetch('http://localhost:8000/api/submit-response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attempt_id: attemptId,  // Include attempt ID
          test_id: testId,        // Include test ID
          user_id: userId,        // Include user ID
          responses,              // Send all responses at once
        }),
      });
  
      // Check if the response is OK (status in the range 200-299)
      if (!response.ok) {
        
        console.error('Failed to submit responses');
        return; // Exit the function if the response is not OK
      }
  
      // Parse the JSON response only if the response is OK
      const data = await response.json();
      console.log("Data after submitting responses to server:", data);
  
      if (data.success) {
        // Navigate to the TestResultPage with the attemptId
        navigate(`/test-result/${attemptId}`);
      } else {
        console.error('Submission unsuccessful:', data.message);
      }
    } catch (error) {
      console.error('Error submitting test:', error);
    }
  };
  
  return (
    <div className="test-interface">
      <div className="question-container">
        {questions.length > 0 && (
          <div className="question-card">
            <div className="test-info">
              <h1>{testName}</h1> {/* Display the test name */}
              <h3>Time Remaining: {formatTime(timer)}</h3> {/* Display countdown timer */}
            </div>
            <h2>Question {currentQuestionIndex + 1}</h2>
            <img src={questions[currentQuestionIndex].question_link} alt={`Question ${currentQuestionIndex + 1}`} className="question-image" />
            {questions[currentQuestionIndex].options.map(option => (
              <div key={option.id} className="option">
                <input
                  type="radio"
                  name={`question-${questions[currentQuestionIndex].id}`}
                  value={option.id}
                  checked={responses[questions[currentQuestionIndex].id] === option.id}
                  onChange={() => handleAnswerChange(questions[currentQuestionIndex].id, option.id)}
                />
                <label>{option.option_text}</label>
              </div>
            ))}
            <div className="navigation-buttons">
              <button className="navigation-button" onClick={() => setCurrentQuestionIndex(prev => Math.max(prev - 1, 0))}>Previous</button>
              <button className="navigation-button" onClick={() => setCurrentQuestionIndex(prev => Math.min(prev + 1, questions.length - 1))}>Next</button>
            </div>
          </div>
        )}
      </div>
      <div className="sidebar">
        {questions.map((question, index) => (
          <div
            key={question.id}
            className={`sidebar-item ${responses[question.id] ? 'answered' : 'not-answered'}`}
            onClick={() => setCurrentQuestionIndex(index)}
          >
            {index + 1}
          </div>
        ))}
        <button onClick={handleSubmit} className="submit-button">Submit</button>
      </div>
    </div>
  );
};

export default TestInterface;