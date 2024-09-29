import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import About from './components/About';
import Services from './components/Services';
import Contact from './components/Contact';
import Login from './components/Login';   
import Register from './components/Register';
import StudentDashboard from './pages/StudentDashboard';
import TeacherDashboard from './pages/TeacherDashboard';
import TestListPage from './pages/TestListPage';
import TestInterface from './pages/TestInterface';
import TestResultPage from './pages/TestResultPage';
import TestAnalysisPage from './pages/TestAnalysisPage';
import AddQuestion from './pages/AddQuestion';
import CreateTest from './pages/CreateTest';
import Layout from './components/Layout';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/services" element={<Services />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/student-dashboard" element={<StudentDashboard />} />
          <Route path="/teacher-dashboard" element={<TeacherDashboard />} />
          <Route path="/tests/:category" element={<TestListPage />} />
          <Route path="/test/:testId" element={<TestInterface />} />
          <Route path="/test-result/:attemptId" element={<TestResultPage />} />
          <Route path="/test-analysis/:attemptId" element={<TestAnalysisPage />} />
          <Route path="/create-test" element={<CreateTest />} />
          <Route path="/test-questionCreation" element={<AddQuestion />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;