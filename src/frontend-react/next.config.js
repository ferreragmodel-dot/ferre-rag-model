/** @type {import('next').NextConfig} */
const nextConfig = {
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
    ],
  },
};

module.exports = nextConfig;
