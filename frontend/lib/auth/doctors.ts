// ─────────────────────────────────────────────────────────────────────────────
// Doctor credential store — SHA-256 hashed passwords (client-side verification)
//
// How passwords are hashed (run once to generate):
//   const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("password"));
//   const hex = Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,'0')).join('');
//
// Seeded accounts:
//   dr.sharma@cardiac.ai     / CardioAI@2025
//   dr.patel@cardiac.ai      / Troponin#99
//   dr.mehta@cardiac.ai      / ECGwave$42
//   admin@cardiac.ai         / Admin@123
// ─────────────────────────────────────────────────────────────────────────────

export interface Doctor {
  id: string;
  email: string;
  passwordHash: string; // SHA-256 hex of plaintext password
  name: string;
  initials: string;
  specialty: string;
  role: "attending" | "resident" | "admin";
  department: string;
}

export const DOCTORS: Doctor[] = [
  {
    id: "doc-001",
    email: "dr.sharma@cardiac.ai",
    // SHA-256("CardioAI@2025")
    passwordHash: "994148c54c1b9c258499393a4e674bcc2fcebe4f0657f3b663468eb69f107735",
    name: "Dr. Arjun Sharma",
    initials: "AS",
    specialty: "Interventional Cardiology",
    role: "attending",
    department: "Cardiology — CATH Lab",
  },
  {
    id: "doc-002",
    email: "dr.patel@cardiac.ai",
    // SHA-256("Troponin#99")
    passwordHash: "b25c06bcbba9350fd4b0785f50380e1118f5e5b71e99457ac5664f099ae7e08a",
    name: "Dr. Priya Patel",
    initials: "PP",
    specialty: "Electrophysiology",
    role: "attending",
    department: "Cardiology — EP Unit",
  },
  {
    id: "doc-003",
    email: "dr.mehta@cardiac.ai",
    // SHA-256("ECGwaves$42")
    passwordHash: "0cef1a9cee9d0bf4b36b51bff5a8d7f6ba5af2dc41eba292239fd330b71c9d57",
    name: "Dr. Rohan Mehta",
    initials: "RM",
    specialty: "Cardiac Imaging & AI",
    role: "resident",
    department: "Cardiology — Research Wing",
  },
  {
    id: "doc-004",
    email: "admin@cardiac.ai",
    // SHA-256("Admin@123")
    passwordHash: "e86f78a8a3caf0b60d8e74e5942aa6d86dc150cd3c03338aef25b7d2d7e3acc7",
    name: "Dr. Pritam Sen",
    initials: "PS",
    specialty: "Cardiology AI Research",
    role: "admin",
    department: "Clinical Decision Support",
  },
];

/** Compute SHA-256 hex of a string using Web Crypto API */
export async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text)
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Verify email + password against the credentials store */
export async function verifyCredentials(
  email: string,
  password: string
): Promise<Doctor | null> {
  const hash = await sha256(password);
  const doctor = DOCTORS.find(
    (d) => d.email.toLowerCase() === email.toLowerCase() && d.passwordHash === hash
  );
  return doctor ?? null;
}
