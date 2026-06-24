import { useState } from "react";

import LiveTherapy from "./pages/LiveTherapy";
import { Landing } from "./components/Landing";
import Sidebar from "./components/Sidebar";

import "./App.css";

export default function App() {
  const [page, setPage] = useState("landing");

  return (
    <div className="app">
      {page !== "landing" && <Sidebar setPage={setPage} page={page} />}

      <div className="content">
        {page === "landing" && <Landing onStart={() => setPage("therapy")} />}
        {page === "therapy" && <LiveTherapy />}
      </div>
    </div>
  );
}