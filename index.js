import express from "express";
import { spawn } from "child_process";
const app = express();
app.use(express.json());

app.post("/", (req, res) => {
    const input = JSON.stringify({ prompt: req.body.prompt });

    const py = spawn("python", ["predict_prompt.py", input]);
    let stdout = "";
    let stderr = "";

     py.stdout.on("data", (data) => {
       stdout += data;
     });

     py.stderr.on("data", (data) => {
       stderr += data;
     });

     py.on("close", (code) => {
       if (code !== 0) {
         return res.status(500).send({ error: stderr || "Prediction failed" });
       }
       res.send({ quality: (stdout.trim()) });
     });
});
app.listen(3000, () => console.log("API running on port 3000"));
