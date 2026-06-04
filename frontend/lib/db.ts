import { Pool } from "pg";

// Singleton pool — reused across hot-reloads in Next.js dev mode
declare global {
  var _pgPool: Pool | undefined;
}

function createPool(): Pool {
  return new Pool({
    host:     process.env.POSTGRES_HOST     || "localhost",
    port:     parseInt(process.env.POSTGRES_PORT || "5432", 10),
    database: process.env.POSTGRES_DB       || "ami_platform",
    user:     process.env.POSTGRES_USER     || "postgres",
    password: process.env.POSTGRES_PASSWORD || "",
    max: 10,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
  });
}

export const pool: Pool = global._pgPool ?? createPool();
if (process.env.NODE_ENV !== "production") global._pgPool = pool;

export async function query<T = Record<string, unknown>>(
  sql: string,
  params?: unknown[]
): Promise<T[]> {
  const client = await pool.connect();
  try {
    const result = await client.query(sql, params);
    return result.rows as T[];
  } finally {
    client.release();
  }
}
