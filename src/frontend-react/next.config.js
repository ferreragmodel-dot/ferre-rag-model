/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: false,
    // rewrites: async () => {
    //     return [
    //         {
    //             source: "/api/:path*",
    //             destination:
    //                 process.env.NODE_ENV === "development"
    //                     ? "http://rag-system-api-service:9000/:path*"
    //                     : "/api/",
    //         },
    //         {
    //             source: "/docs",
    //             destination:
    //                 process.env.NODE_ENV === "development"
    //                     ? "http://rag-system-api-service:9000/docs"
    //                     : "/api/docs",
    //         },
    //         {
    //             source: "/openapi.json",
    //             destination:
    //                 process.env.NODE_ENV === "development"
    //                     ? "http://rag-system-api-service:9000/openapi.json"
    //                     : "/api/openapi.json",
    //         },
    //     ];
    // },
};

module.exports = nextConfig;