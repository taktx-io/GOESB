/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    OESB_API_URL: process.env.OESB_API_URL ?? "http://127.0.0.1:8000",
  },
};
export default nextConfig;
