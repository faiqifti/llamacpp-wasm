// import type { NextConfig } from 'next';

// const nextConfig: NextConfig = {
//   async headers() {
//     return [
//       {
//         source: '/:path*',
//         headers: [
//           {
//             key: 'Cross-Origin-Opener-Policy',
//             value: 'same-origin',
//           },
//           {
//             key: 'Cross-Origin-Embedder-Policy',
//             value: 'require-corp',
//           },
//         ],
//       },
//     ];
//   },

//   // ✅ Disable Turbopack to make @xenova/transformers work
//   experimental: {},
// };

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  webpack: (config, { isServer }) => {
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    config.module.rules.push({
      test: /\.wasm$/,
      type: 'webassembly/async',
    });

    return config;
  },

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
        ],
      },
      {
        source: '/_next/static/chunks/(.*).wasm',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
        ],
      },
    ];
  },

  images: {
    domains: [
      'huggingface.co',
      'cdn-lfs.huggingface.co',
    ],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.huggingface.co',
        pathname: '**',
      },
    ],
  },

  transpilePackages: [
    '@wllama/wllama',
  ],
};

export default nextConfig;