import type { NextConfig } from 'next';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  webpack: (config, { isServer, dev }) => {
    // Enable WebAssembly support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // Add rule for WASM files
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'webassembly/async',
    });

    // Increase memory limits for larger models
    config.performance = {
      ...config.performance,
      maxAssetSize: 1000 * 1024 * 1024, // 1GB
      maxEntrypointSize: 1000 * 1024 * 1024, // 1GB
    };

    // Optimize chunk splitting for better memory usage
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        maxSize: 1000 * 1024, // 1MB chunks
      },
    };

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

  // Disable type checking during build to reduce memory usage
  typescript: {
    ignoreBuildErrors: false,
  },

  eslint: {
    ignoreDuringBuilds: false,
  },

  // Increase timeout for builds
  staticPageGenerationTimeout: 1000,

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