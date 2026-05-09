/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "9000",
        pathname: "/archive/image",
      },
      {
        protocol: "http",
        hostname: "localhost",
        port: "9000",
        pathname: "/design-images/**",
      },
      {
        protocol: "https",
        hostname: "ferre-api-323252296985.us-central1.run.app",
        pathname: "/archive/image",
      },
    ],
  },
};

module.exports = nextConfig;
