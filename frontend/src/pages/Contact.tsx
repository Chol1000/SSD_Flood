import { Link } from "react-router-dom";
import DocPage from "../components/DocPage";

const TOC = [
  { id: "emergency", label: "In Immediate Danger" },
  { id: "coordination", label: "Who Coordinates Response" },
  { id: "prediction-feedback", label: "Feedback on a Prediction" },
];

export default function Contact() {
  return (
    <DocPage eyebrow="In Case of Emergency" title="Contact &amp; Emergency Resources" toc={TOC}>
          <p style={{ fontFamily: "inherit", fontSize: "1.1rem", lineHeight: 1.65, color: "var(--color-text)", margin: "1.4rem 0 0" }}>
            This system is a decision-support tool — it estimates risk, it does not dispatch help. This page is
            about what to do with that distinction: where to direct an actual emergency, and where to send feedback
            about the system itself.
          </p>

          <Rule />

          <h2 id="emergency" style={h2}>If You Are in Immediate Danger</h2>
          <blockquote style={{
            margin: "1.1rem 0", padding: "0.2rem 0 0.2rem 1.1rem", borderLeft: "3px solid var(--risk-critical)",
            fontFamily: "inherit", fontSize: "1.05rem", lineHeight: 1.65, color: "var(--color-text)", fontStyle: "italic",
          }}>
            Do not wait on this website. Move to higher ground immediately, and contact your County Commissioner's
            office, local chief, or community leader — the people who can actually coordinate an evacuation or
            rescue in your area right now.
          </blockquote>
          <p style={bodyText}>
            This page deliberately does not publish a specific emergency phone number. Numbers for local
            authorities, the Relief and Rehabilitation Commission (RRC), and humanitarian responders change and
            vary by county, and a wrong number published here could cost time in a real emergency. Get the current
            number for your area from your county administration, a local radio station, or the official channels
            listed below.
          </p>

          <Rule />

          <h2 id="coordination" style={h2}>Who Coordinates Flood Response in South Sudan</h2>
          <dl style={{ margin: "1.2rem 0 0" }}>
            <Entry
              title="Relief and Rehabilitation Commission (RRC)"
              body="The South Sudanese government body responsible for coordinating disaster relief nationally, working with county administrations on the ground."
            />
            <Entry
              title="County Commissioner's Office"
              body="The first point of contact locally — coordinates evacuation, shelter, and immediate response within a specific county."
            />
            <Entry
              title="UN OCHA South Sudan"
              body="Coordinates the international humanitarian response and publishes situation reports during major flood events."
            />
            <Entry
              title="WFP & IOM South Sudan"
              body="Typically lead on emergency food assistance and displacement/shelter response respectively during flood emergencies."
            />
          </dl>
          <p style={{ ...bodyText, fontSize: "0.86rem", color: "var(--color-text-muted)" }}>
            These are named by role, not with specific contact details, for the same reason noted above — please
            source current contact information from an official government or UN source for your specific location.
          </p>

          <Rule />

          <h2 id="prediction-feedback" style={h2}>What if a Prediction Looks Wrong for My Area?</h2>
          <p style={bodyText}>
            First, check the <Link to="/prediction" style={link}>Prediction</Link> page for your county — the
            "What's Driving This Prediction" panel shows exactly which inputs are live versus historical fallback,
            and the <Link to="/model" style={link}>Model &amp; Validation</Link> page shows precisely how often the
            deployed model is right or wrong on data it never trained on. A single surprising prediction is
            expected sometimes — that's what the published error rates mean in practice.
          </p>
          <p style={bodyText}>
            If you have real, current, on-the-ground information — a flood that's happening right now that the
            model hasn't picked up, for instance — that observation is more valuable relayed to your local RRC or
            County Commissioner's office than reported here. This system informs decision-makers; it doesn't
            replace what people on the ground can see directly.
          </p>

          <Rule />

          <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", lineHeight: 1.7 }}>
            See <Link to="/guide" style={{ color: "var(--color-primary)" }}>How to Use This System</Link> for the bigger
            picture, or <Link to="/about" style={{ color: "var(--color-primary)" }}>About</Link> for full methodology and
            limitations.
          </p>
    </DocPage>
  );
}

function Entry({ title, body }: { title: string; body: string }) {
  return (
    <div style={{ marginBottom: "1.1rem" }}>
      <dt style={{ fontWeight: 700, fontSize: "0.96rem" }}>{title}</dt>
      <dd style={{ margin: "0.25rem 0 0", fontSize: "0.9rem", color: "var(--color-text-muted)", lineHeight: 1.65, maxWidth: 640 }}>{body}</dd>
    </div>
  );
}

const h2: React.CSSProperties = { fontSize: "1.4rem", fontWeight: 700, letterSpacing: "-0.01em", color: "var(--color-text)", marginTop: 0 };
const bodyText: React.CSSProperties = { fontSize: "0.94rem", lineHeight: 1.75, color: "var(--color-text-muted)", marginTop: "1rem", maxWidth: 660 };
const link: React.CSSProperties = { color: "var(--color-primary)", textDecoration: "none", fontWeight: 600 };

function Rule() {
  return <div style={{ height: 1, background: "var(--color-border)", margin: "2.4rem 0" }} />;
}
