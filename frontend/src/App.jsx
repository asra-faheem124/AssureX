import { useEffect, useState } from "react";

function App() {
  const [claim, setClaim] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/claim")
      .then((response) => response.json())
      .then((data) => {
        setClaim(data);
      });
  }, []);

  return (
    <div>
      <h1>AssureX</h1>

      {claim && (
        <div>
          <h2>{claim.product}</h2>

          <p>Fault: {claim.fault}</p>

          <p>
            Warranty Active:{" "}
            {claim.warranty_active ? "Yes" : "No"}
          </p>

          <p>
            Receipt Available:{" "}
            {claim.receipt_available ? "Yes" : "No"}
          </p>

          <p>
            Serial Match:{" "}
            {claim.serial_match ? "Yes" : "No"}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;