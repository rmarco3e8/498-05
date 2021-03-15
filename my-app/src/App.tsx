import './App.css';
import Login from './Views/Login/Login';
import { useEffect, useState } from 'react';

function App() {
  return (
    <>
    { Login && <Login/> }
    </>
  );
}

export default App;
