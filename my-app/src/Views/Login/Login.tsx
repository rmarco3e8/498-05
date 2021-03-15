import { useEffect, useState } from "react";
import './Login.css';

function getWindowDimensions() {
    const { innerWidth: width, innerHeight: height } = window;
    return { width, height };
}

export default function Login(/*props*/){
    const [windowDimensions, setWindowDimensions] = useState(getWindowDimensions());

    useEffect(() => {
        function handleResize() {
        setWindowDimensions(getWindowDimensions());
        }

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    if(windowDimensions.height > windowDimensions.width) return ( 
        <h1 id="msg">
            Please rotate your device to landscape orientation 
            <br/>
            OR 
            <br/>
            Increase the size of your browser
        </h1>
    );

    else return (
        <div className="login-content">
            // return all components for the login view
        </div>
    );
}